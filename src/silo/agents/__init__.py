"""AI agents that trade on the paper book, under the same rules as everything else.

* :mod:`silo.agents.llm` - model clients (a free local model through Ollama,
  or the offline ``rulebot`` used in tests), a call log, and a reply cache.
* :mod:`silo.agents.brief` - turns a :class:`~silo.firewall.PointInTimeView`
  into a short factual brief, with tickers and dates hidden by default.
* :mod:`silo.agents.contestants` - outside designs re-built for Silo:
  a debate desk in the style of TradingAgents and an investing-styles
  panel in the style of ai-hedge-fund.
* :mod:`silo.agents.blend` - Silo's own agent, which combines contestants.
"""

import silo.strategies  # noqa: E402,F401 - load the registry first; it imports this package's agents
