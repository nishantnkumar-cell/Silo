import pytest

from silo.safety import LiveTradingRefused, check_environment, check_order_endpoint


def test_paper_endpoint_allowed():
    assert check_order_endpoint("https://paper-api.alpaca.markets/v2")


@pytest.mark.parametrize(
    "url", ["https://api.alpaca.markets/v2", "https://api.binance.com", "https://example.com/orders"]
)
def test_live_or_unknown_endpoints_refused(url):
    with pytest.raises(LiveTradingRefused):
        check_order_endpoint(url)


def test_live_env_vars_refused():
    with pytest.raises(LiveTradingRefused):
        check_environment({"BROKER_LIVE_API_KEY": "abc"})
    with pytest.raises(LiveTradingRefused):
        check_environment({"APCA_API_BASE_URL": "https://api.alpaca.markets"})
    check_environment({"APCA_API_BASE_URL": "https://paper-api.alpaca.markets"})
