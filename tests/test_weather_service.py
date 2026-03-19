import os
import unittest
from unittest.mock import Mock, patch

import requests

from smart_clean_agent.services.weather_service import format_weather_info, get_weather_by_city


class WeatherServiceTestCase(unittest.TestCase):
    def test_get_weather_by_city_returns_config_message_without_api_key(self):
        with patch.dict(os.environ, {}, clear=True):
            result = get_weather_by_city("北京")

        self.assertEqual(result, "天气服务未配置")

    def test_format_weather_info_returns_stable_text(self):
        result = format_weather_info(
            "北京",
            {
                "weather": "晴",
                "temperature": "26",
                "humidity": "50",
                "winddirection": "北",
                "windpower": "3",
            },
        )

        self.assertIn("城市: 北京", result)
        self.assertIn("天气: 晴", result)
        self.assertIn("温度: 26摄氏度", result)
        self.assertIn("空气质量: 接口未提供", result)

    def test_get_weather_by_city_returns_unavailable_when_request_fails(self):
        mock_session = Mock()
        mock_session.get.side_effect = requests.RequestException("network error")

        with patch.dict(os.environ, {"AMAP_WEATHER_API_KEY": "test-key"}, clear=True):
            result = get_weather_by_city("北京", session=mock_session)

        self.assertEqual(result, "天气服务暂时不可用")


if __name__ == "__main__":
    unittest.main()

