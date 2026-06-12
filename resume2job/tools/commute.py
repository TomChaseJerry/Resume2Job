"""
通勤计算模块（Commute Calculator）

面向「岗位推荐」场景：用户在提问中给出现住址与通勤时间上限
（如「我住在北京五道口，要求实习地点通勤一小时以内」），
系统据此规划检索（城市过滤 + 扩大召回）并对候选岗位计算通勤时长、过滤排序。

分工（与项目既有「LLM 负责语义、Python 负责确定性编排」一致）：
    - LLM   ：仅做通勤意图抽取（extract_commute_intent，对应设计中的 Prompt 1）；
    - Python：高德 API 参数构造（替代 Prompt 2）、地理编码 + 路线规划调用、
              通勤过滤与排序（替代 Prompt 3）、面向用户的通勤摘要文案（模板）。

外部依赖：
    - 高德地图 Web 服务 API，Key 从环境变量 AMAP_API_KEY 读取；
    - 仅用标准库 urllib 发起 HTTP GET，不引入第三方 HTTP 库。
"""

import os
import re
import json
import urllib.parse
import urllib.request
from typing import Optional, Any


# ===== 高德接口常量 =====
AMAP_V3 = "https://restapi.amap.com/v3"
AMAP_V4 = "https://restapi.amap.com/v4"

# 交通方式 -> 路线规划 endpoint（cycling 走 v4 骑行接口，其余走 v3）
TRANSPORT_ENDPOINT = {
    "transit": (AMAP_V3, "direction/transit/integrated"),
    "driving": (AMAP_V3, "direction/driving"),
    "walking": (AMAP_V3, "direction/walking"),
    "cycling": (AMAP_V4, "direction/bicycling"),
}
VALID_TRANSPORT = set(TRANSPORT_ENDPOINT.keys())
DEFAULT_TRANSPORT = "transit"

# 直辖市（地址中无「XX市」时兜底识别城市）
_MUNICIPALITIES = ["北京", "上海", "天津", "重庆"]

# LLM 接口（意图抽取，统一走 core 层）
from resume2job.core.config import CHAT_MODEL as MODEL_NAME
from resume2job.core.llm import call_llm as _core_call_llm, clean_llm_json_output as _clean_json

HTTP_TIMEOUT = 6  # 秒


# ===== 工具函数 =====
def extract_city(address: Optional[str]) -> Optional[str]:
    """从地址文本中粗提取城市名（用于检索 city_filter 与高德 city 参数）。"""
    if not isinstance(address, str) or not address.strip():
        return None
    m = re.search(r"([一-龥]{2,8}?市)", address)
    if m:
        return m.group(1)
    for c in _MUNICIPALITIES:
        if c in address:
            return c
    return None


def _normalize_transport(value: Any) -> str:
    """交通方式归一化，非法 / 缺失时回退默认 transit。"""
    if isinstance(value, str) and value.strip().lower() in VALID_TRANSPORT:
        return value.strip().lower()
    return DEFAULT_TRANSPORT


# ===== Prompt 1：通勤意图抽取（LLM）=====
_COMMUTE_INTENT_SYSTEM = """你是一个求职助手中的意图解析模块。你的任务是从用户问题中提取通勤相关约束信息。

## 抽取规则
1. 识别用户当前居住地址（越具体越好，至少到区级）；
2. 识别通勤时间上限（如「1小时以内」「45分钟」），统一换算为分钟（「1小时」→60，「1.5小时」→90）；
3. 识别交通方式偏好（transit 地铁/公交 / driving 驾车 / walking 步行 / cycling 骑行），未提及默认 transit；
4. 若用户只给大概位置（如「朝阳区」），记为模糊地址，不得自行补全。

## 处理规则
- 若用户未提供住址，user_address 返回 null，has_commute_constraint 返回 false；
- 缺失字段一律返回 null，禁止猜测；
- address_confidence 取值：exact（具体到门牌）/ district（到区或商圈）/ vague（仅城市或更模糊）。

## 输出（严格 JSON，不要输出任何额外文字或 Markdown）
{
  "has_commute_constraint": true,
  "user_address": "北京市海淀区五道口",
  "address_confidence": "district",
  "max_commute_minutes": 60,
  "preferred_transport": "transit",
  "raw_address_text": "北京五道口"
}"""


def _empty_intent(error: Optional[str] = None) -> dict:
    """无约束 / 失败时的默认意图结构。"""
    return {
        "has_commute_constraint": False,
        "user_address": None,
        "address_confidence": None,
        "max_commute_minutes": None,
        "preferred_transport": None,
        "raw_address_text": None,
        "error": error,
    }


def _validate_intent(data: Any) -> dict:
    """对 LLM 抽取结果做类型校验与兜底。"""
    if not isinstance(data, dict):
        return _empty_intent("通勤意图输出非法")

    address = data.get("user_address")
    address = address.strip() if isinstance(address, str) and address.strip() else None

    has_constraint = bool(data.get("has_commute_constraint")) and address is not None

    minutes = data.get("max_commute_minutes")
    if isinstance(minutes, bool):
        minutes = None
    elif isinstance(minutes, (int, float)):
        minutes = int(round(minutes))
    elif isinstance(minutes, str):
        m = re.search(r"\d+", minutes)
        minutes = int(m.group()) if m else None
    else:
        minutes = None

    confidence = data.get("address_confidence")
    if confidence not in {"exact", "district", "vague"}:
        confidence = None

    transport = data.get("preferred_transport")
    transport = transport.strip().lower() if isinstance(transport, str) and transport.strip() else None
    if transport is not None and transport not in VALID_TRANSPORT:
        transport = None

    raw_addr = data.get("raw_address_text")
    raw_addr = raw_addr.strip() if isinstance(raw_addr, str) and raw_addr.strip() else None

    return {
        "has_commute_constraint": has_constraint,
        "user_address": address,
        "address_confidence": confidence,
        "max_commute_minutes": minutes,
        "preferred_transport": transport,
        "raw_address_text": raw_addr,
        "error": None,
    }


def extract_commute_intent(user_query: str) -> dict:
    """Prompt 1：用 LLM 从用户问题抽取通勤约束。失败时返回无约束结构 + error。"""
    if not isinstance(user_query, str) or not user_query.strip():
        return _empty_intent("用户问题为空")

    try:
        raw = _core_call_llm(
            _COMMUTE_INTENT_SYSTEM,
            f"## 用户输入\n{user_query}\n\n请按规则输出 JSON。",
            model=MODEL_NAME,
        )
        data = json.loads(_clean_json(raw))
    except Exception as exc:  # 缺 Key / 网络 / SDK / JSON 解析等
        return _empty_intent(f"通勤意图抽取失败：{exc}")

    return _validate_intent(data)


# ===== 高德 HTTP 调用（Python，替代 Prompt 2 的「参数生成」）=====
def _http_get_json(base: str, path: str, params: dict) -> Optional[dict]:
    """发起一次高德 GET 请求并解析 JSON。任何异常返回 None。"""
    try:
        qs = urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
        url = f"{base}/{path}?{qs}"
        req = urllib.request.Request(url, headers={"User-Agent": "Resume2Job/1.0"})
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


def amap_geocode(address: Optional[str], city: Optional[str], api_key: str, cache: dict) -> Optional[str]:
    """地理编码：地址文字 -> "lng,lat"。带进程内缓存；失败返回 None。"""
    if not isinstance(address, str) or not address.strip():
        return None
    ck = (address.strip(), city or "")
    if ck in cache:
        return cache[ck]

    result = None
    data = _http_get_json(AMAP_V3, "geocode/geo",
                          {"key": api_key, "address": address.strip(), "city": city or ""})
    if data and data.get("status") == "1" and data.get("geocodes"):
        loc = data["geocodes"][0].get("location")
        if isinstance(loc, str) and "," in loc:
            result = loc
    cache[ck] = result
    return result


def _parse_duration_seconds(data: Optional[dict], base: str) -> Optional[int]:
    """从高德路线规划响应中解析时长（秒）。结构因接口而异。"""
    if not isinstance(data, dict):
        return None
    try:
        if base == AMAP_V4:
            # 骑行 v4：{"errcode":0,"data":{"paths":[{"duration":...}]}}
            if data.get("errcode") not in (0, "0"):
                return None
            paths = (data.get("data") or {}).get("paths") or []
            if paths:
                return int(paths[0].get("duration"))
            return None
        # v3
        if data.get("status") != "1":
            return None
        route = data.get("route") or {}
        if "transits" in route:  # 公交/地铁
            transits = route.get("transits") or []
            if transits:
                return int(transits[0].get("duration"))
            return None
        paths = route.get("paths") or []  # 驾车 / 步行
        if paths:
            return int(paths[0].get("duration"))
    except (ValueError, TypeError, AttributeError):
        return None
    return None


def amap_route_minutes(origin: str, destination: str, transport: str,
                       city: Optional[str], api_key: str) -> Optional[int]:
    """调用高德路线规划，返回通勤分钟数（四舍五入）；失败返回 None。"""
    base, path = TRANSPORT_ENDPOINT.get(transport, TRANSPORT_ENDPOINT[DEFAULT_TRANSPORT])
    params = {"key": api_key, "origin": origin, "destination": destination}
    if transport == "transit":
        # 公交接口需要起点城市与终点城市
        params["city"] = city or ""
        params["cityd"] = city or ""
    data = _http_get_json(base, path, params)
    seconds = _parse_duration_seconds(data, base)
    if seconds is None:
        return None
    return max(0, round(seconds / 60))


# ===== 通勤摘要文案（Python 模板，替代 Prompt 3 的自然语言生成）=====
_TRANSPORT_CN = {"transit": "地铁/公交", "driving": "驾车", "walking": "步行", "cycling": "骑行"}


def _commute_summary(minutes: Optional[int], within: Optional[bool],
                     transport: str, limit: Optional[int], addr_known: bool) -> str:
    """生成面向用户的通勤说明，不暴露 API 字段名。"""
    mode_cn = _TRANSPORT_CN.get(transport, "地铁/公交")
    if not addr_known:
        return "办公地址未知，无法计算通勤"
    if minutes is None:
        return f"{mode_cn}路线无法计算"
    if within:
        return f"{mode_cn}约 {minutes} 分钟，符合通勤要求"
    if limit:
        return f"{mode_cn}约 {minutes} 分钟，超出你设定的 {limit} 分钟上限"
    return f"{mode_cn}约 {minutes} 分钟"


# ===== Prompt 3 等价逻辑：过滤与排序（Python）=====
def _rank_jobs(records: list, max_minutes: Optional[int], transport: str) -> dict:
    """对带通勤时长的岗位做过滤与排序。

    records：[{job_id, company, title, final_score, commute_minutes(None=不可计算), addr_known}]
    规则：
      - 达标（在上限内）按 final_score 降序排前；
      - 超标的不丢弃，排到末位；
      - 若达标不足 3 个，从超标里按分数补足并注明超标；
      - 地址缺失 / 无法计算的 within_limit=None，排最后。
    """
    def _score(r):
        return r.get("final_score") or 0

    available = [r for r in records if r.get("addr_known") and r.get("commute_minutes") is not None]
    unknown = [r for r in records if not (r.get("addr_known") and r.get("commute_minutes") is not None)]

    if max_minutes is None:
        # 没有明确上限：全部按通勤时长升序（可计算的在前），不做达标判定
        within = sorted(available, key=lambda r: r["commute_minutes"])
        over = []
    else:
        within = sorted([r for r in available if r["commute_minutes"] <= max_minutes],
                        key=_score, reverse=True)
        over = sorted([r for r in available if r["commute_minutes"] > max_minutes],
                      key=_score, reverse=True)

    # 达标不足 3 个时，从超标中按分数补足
    backfilled = []
    if max_minutes is not None and len(within) < 3 and over:
        need = 3 - len(within)
        backfilled = over[:need]
        over = over[need:]

    ranked_records = within + backfilled
    filtered_records = over + unknown

    def _to_view(r, rank=None, mark_within=None):
        minutes = r.get("commute_minutes")
        addr_known = r.get("addr_known")
        if not addr_known or minutes is None:
            within_limit = None
        elif max_minutes is None:
            within_limit = True
        else:
            within_limit = minutes <= max_minutes
        view = {
            "job_id": r.get("job_id"),
            "company": r.get("company"),
            "title": r.get("title"),
            "commute_time_minutes": minutes,
            "within_limit": within_limit,
            "commute_summary": _commute_summary(minutes, within_limit, transport, max_minutes, bool(addr_known)),
            "final_score": r.get("final_score"),
        }
        if rank is not None:
            view["rank"] = rank
        return view

    ranked_jobs = [_to_view(r, rank=i + 1) for i, r in enumerate(ranked_records)]
    filtered_out = []
    for r in filtered_records:
        v = _to_view(r)
        v["reason_excluded"] = ("通勤时间超标" if (r.get("addr_known") and r.get("commute_minutes") is not None)
                                else "办公地址未知，无法计算通勤")
        filtered_out.append(v)

    total = len(records)
    qualified = len(within)
    note_parts = [f"共检索 {total} 个候选岗位"]
    if max_minutes is not None:
        note_parts.append(f"{qualified} 个通勤达标")
        if backfilled:
            note_parts.append(f"{len(backfilled)} 个虽超标但因达标不足已补充")
        if over:
            note_parts.append(f"{len(over)} 个超标排至末位")
    if unknown:
        note_parts.append(f"{len(unknown)} 个地址缺失无法计算")
    note = "，".join(note_parts) + "。"

    return {"ranked_jobs": ranked_jobs, "filtered_out": filtered_out, "note": note}


# ===== 对外主入口：计算 + 排序 =====
def compute_and_rank(intent: dict, jobs: list) -> dict:
    """根据通勤意图与候选岗位，计算各岗位通勤时长并过滤排序。

    jobs：[{job_id, company, title, office_address, city, final_score}]
    返回：{ranked_jobs, filtered_out, note, per_job, error}
      - per_job：{job_id -> {commute_time_minutes, within_limit, commute_summary, transport}}
    """
    intent = intent or {}
    transport = _normalize_transport(intent.get("preferred_transport"))
    max_minutes = intent.get("max_commute_minutes")
    user_address = intent.get("user_address")

    api_key = os.environ.get("AMAP_API_KEY")
    error = None
    origin = None

    if not api_key:
        error = "未配置 AMAP_API_KEY，跳过通勤计算"
    elif not user_address:
        error = "缺少用户住址，跳过通勤计算"
    else:
        cache: dict = {}
        origin_city = extract_city(user_address)
        origin = amap_geocode(user_address, origin_city, api_key, cache)
        if not origin:
            error = f"用户住址地理编码失败：{user_address}"

    # 逐岗位计算（若前置失败，则全部记为不可计算）
    records = []
    geocode_cache: dict = {}
    for job in jobs or []:
        office = job.get("office_address")
        job_city = job.get("city") or extract_city(office)
        minutes = None
        addr_known = bool(office)

        if origin and api_key and addr_known:
            dest = amap_geocode(office, job_city, api_key, geocode_cache)
            if dest:
                minutes = amap_route_minutes(origin, dest, transport, job_city, api_key)
            else:
                addr_known = False  # 目的地无法编码，等同地址不可用

        records.append({
            "job_id": job.get("job_id"),
            "company": job.get("company"),
            "title": job.get("title"),
            "final_score": job.get("final_score"),
            "commute_minutes": minutes,
            "addr_known": addr_known,
        })

    ranked = _rank_jobs(records, max_minutes, transport)

    # 构建 per_job 便于节点回填到 match_results
    per_job = {}
    for v in ranked["ranked_jobs"] + ranked["filtered_out"]:
        per_job[v["job_id"]] = {
            "commute_time_minutes": v["commute_time_minutes"],
            "within_limit": v["within_limit"],
            "commute_summary": v["commute_summary"],
            "transport": transport,
        }

    ranked["per_job"] = per_job
    ranked["error"] = error
    return ranked


# ===== 测试入口 =====
# 本模块提供两类测试：
#   1) offline_test（默认，python commute_calculator.py）：
#      只对「确定性、可离线验证」的纯函数做断言，不发起任何网络请求。
#   2) agent_flow_test（python commute_calculator.py agent）：
#      端到端验证 Agent 流程——
#      用户自然语言问题 → LLM 判断通勤意图（Prompt 1）
#      → 高德地图 API 计算各候选岗位通勤时长 → 体现到岗位召回（过滤 + 重排）。
#      需联网且已配置 DASHSCOPE_API_KEY 与 AMAP_API_KEY。
def offline_test():
    """离线自测：python commute_calculator.py

    覆盖城市提取、交通方式归一化、LLM 输出清洗与校验、高德响应时长解析、
    通勤摘要文案、过滤排序，以及无 AMAP_API_KEY 时 compute_and_rank 的兜底。
    """
    passed = 0
    failed = 0

    def check(name, got, want):
        nonlocal passed, failed
        if got == want:
            passed += 1
            print(f"  [PASS] {name}")
        else:
            failed += 1
            print(f"  [FAIL] {name}\n         got : {got!r}\n         want: {want!r}")

    print("== extract_city ==")
    check("含「市」", extract_city("北京市海淀区五道口"), "北京市")
    check("直辖市兜底", extract_city("北京五道口"), "北京")
    check("空地址", extract_city("   "), None)
    check("非字符串", extract_city(None), None)

    print("== _normalize_transport ==")
    check("合法值", _normalize_transport("DRIVING"), "driving")
    check("非法回退", _normalize_transport("飞行"), DEFAULT_TRANSPORT)
    check("缺失回退", _normalize_transport(None), DEFAULT_TRANSPORT)

    print("== _clean_json ==")
    check("剥离代码块", _clean_json('```json\n{"a": 1}\n```'), '{"a": 1}')
    check("截取首尾花括号", _clean_json('前缀 {"a": 1} 后缀'), '{"a": 1}')
    check("空串", _clean_json(""), "")

    print("== _validate_intent ==")
    valid = _validate_intent({
        "has_commute_constraint": True,
        "user_address": "北京市海淀区五道口",
        "address_confidence": "district",
        "max_commute_minutes": "60分钟",
        "preferred_transport": "TRANSIT",
        "raw_address_text": "北京五道口",
    })
    check("地址", valid["user_address"], "北京市海淀区五道口")
    check("约束成立", valid["has_commute_constraint"], True)
    check("分钟从字符串提取", valid["max_commute_minutes"], 60)
    check("交通方式小写", valid["preferred_transport"], "transit")
    check("布尔分钟视为无效", _validate_intent({"max_commute_minutes": True})["max_commute_minutes"], None)
    check("无地址则约束不成立",
          _validate_intent({"has_commute_constraint": True, "user_address": "  "})["has_commute_constraint"], False)
    check("非法置信度归零", _validate_intent({"address_confidence": "高"})["address_confidence"], None)
    check("非 dict 输入", _validate_intent("x")["error"], "通勤意图输出非法")

    print("== _parse_duration_seconds ==")
    check("v3 公交", _parse_duration_seconds(
        {"status": "1", "route": {"transits": [{"duration": "1800"}]}}, AMAP_V3), 1800)
    check("v3 驾车", _parse_duration_seconds(
        {"status": "1", "route": {"paths": [{"duration": "900"}]}}, AMAP_V3), 900)
    check("v3 失败状态", _parse_duration_seconds({"status": "0"}, AMAP_V3), None)
    check("v4 骑行", _parse_duration_seconds(
        {"errcode": 0, "data": {"paths": [{"duration": 1200}]}}, AMAP_V4), 1200)
    check("v4 错误码", _parse_duration_seconds({"errcode": 1}, AMAP_V4), None)
    check("非 dict", _parse_duration_seconds(None, AMAP_V3), None)

    print("== _commute_summary ==")
    check("地址未知", _commute_summary(None, None, "transit", 60, False), "办公地址未知，无法计算通勤")
    check("无法计算", _commute_summary(None, None, "driving", 60, True), "驾车路线无法计算")
    check("达标", _commute_summary(30, True, "transit", 60, True), "地铁/公交约 30 分钟，符合通勤要求")
    check("超标", _commute_summary(80, False, "cycling", 60, True), "骑行约 80 分钟，超出你设定的 60 分钟上限")

    print("== _rank_jobs ==")
    records = [
        {"job_id": "A", "company": "甲", "title": "实习", "final_score": 0.9, "commute_minutes": 30, "addr_known": True},
        {"job_id": "B", "company": "乙", "title": "实习", "final_score": 0.8, "commute_minutes": 80, "addr_known": True},
        {"job_id": "C", "company": "丙", "title": "实习", "final_score": 0.7, "commute_minutes": None, "addr_known": False},
    ]
    ranked = _rank_jobs(records, 60, "transit")
    check("达标排首位", ranked["ranked_jobs"][0]["job_id"], "A")
    check("超标补足进入 ranked（不足3个）", [j["job_id"] for j in ranked["ranked_jobs"]], ["A", "B"])
    check("地址未知被过滤", ranked["filtered_out"][0]["job_id"], "C")
    check("过滤原因", ranked["filtered_out"][0]["reason_excluded"], "办公地址未知，无法计算通勤")

    print("== compute_and_rank（无 AMAP_API_KEY 兜底）==")
    saved_key = os.environ.pop("AMAP_API_KEY", None)
    try:
        out = compute_and_rank(
            {"user_address": "北京市海淀区", "max_commute_minutes": 60, "preferred_transport": "transit"},
            [{"job_id": "A", "company": "甲", "title": "实习", "office_address": "北京市朝阳区", "final_score": 0.9}],
        )
        check("缺 Key 时返回 error", out["error"], "未配置 AMAP_API_KEY，跳过通勤计算")
        check("岗位仍被收录（不可计算）", "A" in out["per_job"], True)
        check("不可计算 within_limit 为 None", out["per_job"]["A"]["within_limit"], None)
    finally:
        if saved_key is not None:
            os.environ["AMAP_API_KEY"] = saved_key

    print(f"\n结果：{passed} 通过 / {failed} 失败")
    return 0 if failed == 0 else 1


# 模拟「岗位召回」结果：覆盖近 → 远的真实北京地址，供高德实算后做过滤排序。
# 选址刻意拉开通勤梯度（中关村 ≈ 邻近，亦庄经济开发区 ≈ 远郊），
# 以便稳定验证「近的岗位优先、远的被判为超标」这一召回体现，不依赖具体分钟数。
_AGENT_FLOW_SAMPLE_JOBS = [
    {"job_id": "J1", "company": "中关村科技", "title": "数据分析实习",
     "office_address": "北京市海淀区中关村", "city": "北京", "final_score": 0.80},
    {"job_id": "J2", "company": "上地网络", "title": "后端开发实习",
     "office_address": "北京市海淀区上地", "city": "北京", "final_score": 0.85},
    {"job_id": "J3", "company": "国贸资本", "title": "算法实习",
     "office_address": "北京市朝阳区国贸", "city": "北京", "final_score": 0.90},
    {"job_id": "J4", "company": "亦庄智造", "title": "测试开发实习",
     "office_address": "北京市大兴区亦庄经济开发区", "city": "北京", "final_score": 0.95},
]

_AGENT_FLOW_QUERY = "我现住址北京市海淀区五道口，我想找通勤时间小于半小时的实习职位"


def agent_flow_test(user_query: Optional[str] = None) -> int:
    """端到端验证 Agent 通勤流程（需联网 + 两个 API Key）。

    1) LLM 判断意图：从自然语言问题抽取住址 / 时间上限 / 交通方式；
    2) 高德 API 计算：对召回的候选岗位逐一地理编码 + 路线规划；
    3) 体现到召回：按通勤约束过滤 + 重排，达标优先、超标排后。

    断言聚焦「流程不变量」（意图识别正确、高德确被调用、近 < 远、远者超标、
    召回被重排），不对具体分钟数做脆弱断言（高德实时路况会浮动）。
    """
    user_query = user_query or _AGENT_FLOW_QUERY
    passed = 0
    failed = 0
    skipped = False

    def check(name, ok, detail=""):
        nonlocal passed, failed
        if ok:
            passed += 1
            print(f"  [PASS] {name}{('  — ' + detail) if detail else ''}")
        else:
            failed += 1
            print(f"  [FAIL] {name}{('  — ' + detail) if detail else ''}")

    # 前置检查：缺 Key 时跳过而非失败（避免在无凭证环境下误报）
    if not os.environ.get("DASHSCOPE_API_KEY") or not os.environ.get("AMAP_API_KEY"):
        print("[SKIP] 未配置 DASHSCOPE_API_KEY / AMAP_API_KEY，跳过 Agent 流程测试。")
        return 0

    print("=" * 60)
    print("【Agent 通勤流程端到端验证】")
    print(f"用户问题：{user_query}")
    print("=" * 60)

    # ---- 步骤 1：LLM 判断通勤意图（Prompt 1）----
    print("\n[1/3] LLM 抽取通勤意图 ...")
    intent = extract_commute_intent(user_query)
    print(f"      意图：has_constraint={intent.get('has_commute_constraint')}，"
          f"住址={intent.get('user_address')}，"
          f"上限={intent.get('max_commute_minutes')} 分钟，"
          f"交通={intent.get('preferred_transport')}")
    if intent.get("error"):
        print(f"[SKIP] 意图抽取失败：{intent['error']}（多为网络 / 凭证问题），跳过后续。")
        return 0

    check("LLM 识别出通勤约束", intent.get("has_commute_constraint") is True)
    check("时间上限解析为 30 分钟（半小时）", intent.get("max_commute_minutes") == 30,
          f"got={intent.get('max_commute_minutes')}")
    check("交通方式默认/识别为 transit", intent.get("preferred_transport") in (None, "transit"),
          f"got={intent.get('preferred_transport')}")
    check("住址含「海淀」", isinstance(intent.get("user_address"), str)
          and "海淀" in (intent.get("user_address") or ""), intent.get("user_address"))

    # ---- 步骤 2 + 3：高德实算 → 过滤 + 重排（体现到召回）----
    print(f"\n[2/3] 高德 API 计算 {len(_AGENT_FLOW_SAMPLE_JOBS)} 个召回岗位的通勤时长 ...")
    # 高德免费配额有 QPS 限制，突发请求可能被限流（部分岗位算不出）。
    # 测试里做几次「整体重试」，取算出岗位最多的一次，让演示更稳定；生产代码不受影响。
    import time
    total_jobs = len(_AGENT_FLOW_SAMPLE_JOBS)
    result = None
    for attempt in range(1, 4):
        candidate = compute_and_rank(intent, _AGENT_FLOW_SAMPLE_JOBS)
        views = candidate["ranked_jobs"] + candidate["filtered_out"]
        n_ok = sum(1 for v in views if v["commute_time_minutes"] is not None)
        if result is None or n_ok > result[1]:
            result = (candidate, n_ok)
        if n_ok >= total_jobs or candidate.get("error"):
            break
        print(f"      第 {attempt} 次：{n_ok}/{total_jobs} 个算出（疑似限流），稍后重试 ...")
        time.sleep(1.5)
    result = result[0]

    if result.get("error"):
        print(f"[SKIP] 通勤计算未完成：{result['error']}，跳过召回断言。")
        return 0 if failed == 0 else 1

    by_id = {v["job_id"]: v for v in result["ranked_jobs"] + result["filtered_out"]}

    print("\n[3/3] 通勤结果体现到岗位召回：")
    print("      —— 召回排序（ranked_jobs，达标/补足优先）——")
    for v in result["ranked_jobs"]:
        print(f"        #{v.get('rank')} {v['job_id']} {by_id[v['job_id']].get('company','')}"
              f" | {v['commute_summary']}")
    print("      —— 排除/末位（filtered_out）——")
    for v in result["filtered_out"]:
        print(f"        {v['job_id']} | {v['commute_summary']} | {v.get('reason_excluded')}")
    print(f"      召回说明：{result['note']}")

    # 流程不变量断言（稳健，不依赖具体分钟）
    print("\n[断言] 流程不变量：")
    computed = {jid: by_id[jid]["commute_time_minutes"]
                for jid in ("J1", "J2", "J3", "J4")
                if by_id.get(jid) and by_id[jid]["commute_time_minutes"] is not None}
    check("高德确被调用：至少 1 个岗位算出通勤时长", len(computed) >= 1,
          f"computed={computed}")

    near, far = by_id.get("J1"), by_id.get("J4")
    if near and far and near["commute_time_minutes"] is not None and far["commute_time_minutes"] is not None:
        check("距离单调：中关村(近) 通勤 < 亦庄(远) 通勤",
              near["commute_time_minutes"] < far["commute_time_minutes"],
              f"中关村={near['commute_time_minutes']}min < 亦庄={far['commute_time_minutes']}min")
        check("远郊岗位被判为超标（within_limit=False）", far["within_limit"] is False,
              f"亦庄 within_limit={far['within_limit']}")

    qualified = [v for v in result["ranked_jobs"] if v["within_limit"] is True]
    check("召回被重排：ranked_jobs 非空（达标优先，不足则按分数补足）",
          len(result["ranked_jobs"]) >= 1,
          f"达标 {len(qualified)} 个 / 召回 {len(result['ranked_jobs'])} 个")
    check("每条召回都生成了通勤摘要文案",
          all(by_id[jid].get("commute_summary") for jid in by_id))

    print(f"\n结果：{passed} 通过 / {failed} 失败")
    return 0 if failed == 0 else 1


def main(argv=None):
    """入口分发：无参数跑离线断言；`agent` 跑端到端 Agent 流程。"""
    import sys
    argv = sys.argv[1:] if argv is None else argv
    # Windows 控制台默认 GBK，统一切到 UTF-8 以正常显示中文输出
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    mode = argv[0].lower() if argv else "offline"
    if mode in ("agent", "flow", "online", "--agent"):
        query = argv[1] if len(argv) > 1 else None
        return agent_flow_test(query)
    return offline_test()


if __name__ == "__main__":
    import sys
    sys.exit(main())
