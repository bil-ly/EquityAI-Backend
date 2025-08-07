EquityAI - AI Investment Assistant
Vision Statement
Personal AI-powered investment assistant that connects to EasyEquities and uses Claude API to analyze dividend stocks, providing intelligent buy/sell recommendations to optimize portfolio performance.
Core Concept
Problem: Making smart dividend investment decisions requires analyzing complex financial data and market trends
Solution: AI assistant that does the heavy lifting - analyzes your actual portfolio and provides personalized recommendations
Goal: Start investing smarter immediately, compound returns through better decisions
Technical Architecture
Backend (FastAPI)

Authentication: Connect to EasyEquities account
Data Sources:

EasyEquities API (via existing Python client)
Yahoo Finance API (JSE stocks with .JO suffix)
Alpha Vantage (backup financial data)


AI Integration: Claude API for investment analysis
Core Services:

Portfolio data fetching
Dividend history extraction
Real-time stock analysis
Buy/sell recommendation engine



Frontend (Flutter)

Dashboard: Portfolio overview, performance metrics
AI Analysis: Stock-specific Claude insights
Dividend Calendar: Track upcoming payments
Market Scanner: Find new opportunities
Push Notifications: Dividend alerts, market updates

Key Features
MVP Features (Personal Use)

 Connect to EasyEquities account
 Fetch current portfolio holdings
 Extract dividend payment history
 Claude-powered stock analysis
 Buy/sell recommendations
 Portfolio performance tracking

Future Enhancements

 Real-time price alerts
 Automated dividend calendar
 Market opportunity scanner
 Performance benchmarking
 Tax optimization suggestions

Claude Analysis Capabilities

Dividend Sustainability: Analyze payout ratios, cash flow, debt levels
Investment Timing: Market conditions, valuation metrics, technical analysis
Portfolio Optimization: Diversification, risk assessment, rebalancing suggestions
Risk Analysis: Company fundamentals, sector trends, economic factors
Personalized Advice: Based on your actual holdings and investment goals

User Experience Flow

Setup: Login with EasyEquities credentials
Sync: Import current portfolio and transaction history
Analyze: Get AI insights on each holding
Decide: Receive buy/sell/hold recommendations
Track: Monitor performance and receive alerts
Optimize: Continuous AI-driven portfolio improvements

Development Phases
Phase 1: Foundation (Week 1)

FastAPI backend with EasyEquities integration
Basic Claude API analysis
Core data models and endpoints

Phase 2: Intelligence (Week 2)

Advanced dividend analysis algorithms
Sophisticated Claude prompts
Portfolio optimization logic

Phase 3: Interface (Week 3)

Flutter app with key screens
Real-time data updates
Push notification system

Phase 4: Polish (Week 4)

Error handling and edge cases
Performance optimization
User experience refinement

Investment Philosophy
Focus: Dividend growth investing in JSE-listed companies
Strategy: AI-assisted fundamental analysis + technical timing
Risk Management: Diversification across sectors, position sizing
Time Horizon: Long-term wealth building through compound dividend growth
Legal & Compliance

Current Scope: Personal use only
Disclaimers: Educational tool, not licensed financial advice
Future Consideration: FSP consultation if expanding to public users
Data Privacy: Secure handling of personal financial information

Success Metrics

Investment Performance: Beat JSE All Share Index returns
Dividend Growth: Increase annual dividend income by X%
Decision Quality: Reduce emotional/impulsive trades
Time Efficiency: Faster research and analysis
Knowledge Growth: Better understanding of market dynamics

Branding
Name: EquityAI
Tagline: "Intelligent Equity Analysis"
Target User: South African retail investors using EasyEquities
Value Prop: "Make smarter dividend investments with AI analysis"
Risk Mitigation

Technical: Multiple data sources, fallback systems
Financial: Clear disclaimers, encourage diversification
Legal: Personal use only, no unlicensed advice
Operational: Graceful degradation, error handling


"The best time to plant a tree was 20 years ago. The second best time is now."
Let's build the tool that helps you invest smarter, starting today.