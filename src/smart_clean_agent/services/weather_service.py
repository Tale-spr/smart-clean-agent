import os

import requests

from smart_clean_agent.utils.logger_handler import logger

AMAP_GEOCODE_URL = "https://restapi.amap.com/v3/geocode/geo"
AMAP_WEATHER_URL = "https://restapi.amap.com/v3/weather/weatherInfo"
DEFAULT_TIMEOUT = 10


def format_weather_info(city: str, live_info: dict[str, str]) -> str:
    weather = live_info.get("weather", "未知")
    temperature = live_info.get("temperature", "未知")
    humidity = live_info.get("humidity", "未知")
    wind_direction = live_info.get("winddirection", "未知")
    wind_power = live_info.get("windpower", "未知")

    return (
        f"城市: {city}\n"
        f"天气: {weather}\n"
        f"温度: {temperature}摄氏度\n"
        f"湿度: {humidity}%\n"
        f"风向风力: {wind_direction}风 {wind_power}级\n"
        "空气质量: 接口未提供"
    )


def _get_json(url: str, params: dict[str, str], session: requests.Session, timeout: int) -> dict:
    response = session.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    return response.json()


def resolve_city_adcode(city: str, api_key: str, session: requests.Session, timeout: int) -> str | None:
    data = _get_json(
        AMAP_GEOCODE_URL,
        {"address": city, "key": api_key},
        session,
        timeout,
    )
    if data.get("status") != "1":
        return None

    geocodes = data.get("geocodes") or []
    if not geocodes:
        return None

    return geocodes[0].get("adcode")


def fetch_live_weather(adcode: str, api_key: str, session: requests.Session, timeout: int) -> dict[str, str] | None:
    data = _get_json(
        AMAP_WEATHER_URL,
        {"city": adcode, "key": api_key, "extensions": "base"},
        session,
        timeout,
    )
    if data.get("status") != "1":
        return None

    lives = data.get("lives") or []
    if not lives:
        return None

    return lives[0]


def get_weather_by_city(city: str, session: requests.Session | None = None, timeout: int = DEFAULT_TIMEOUT) -> str:
    city = city.strip()
    if not city:
        return "未提供天气查询城市"

    api_key = os.getenv("AMAP_WEATHER_API_KEY", "").strip()
    if not api_key:
        return "天气服务未配置"

    http_session = session or requests.Session()
    try:
        adcode = resolve_city_adcode(city, api_key, http_session, timeout)
        if not adcode:
            return f"未查询到{city}的天气信息"

        live_info = fetch_live_weather(adcode, api_key, http_session, timeout)
        if not live_info:
            return f"未查询到{city}的天气信息"

        return format_weather_info(city, live_info)
    except requests.RequestException as exc:
        logger.warning(f"[天气服务]查询{city}天气失败: {str(exc)}")
        return "天气服务暂时不可用"

