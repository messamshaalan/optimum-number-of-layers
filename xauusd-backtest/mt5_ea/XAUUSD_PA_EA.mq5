//+------------------------------------------------------------------+
//| XAUUSD Price Action EA                                           |
//| Ported from Python backtester (xauusd-backtest)                  |
//| Timeframe: M5 chart  |  Instrument: XAUUSD                       |
//|                                                                  |
//| Strategy: Multi-timeframe price-action gating                    |
//|   - Pin bar and/or engulfing pattern on 15M / 5M / 1M            |
//|   - EMA50 trend filter on 15M                                     |
//|   - ADX regime detection on 15M (TRENDING / RANGING / TRANSIT)   |
//|   - Session-aware trade limits and cooldown periods               |
//|   - NY and Rollover sessions blocked                              |
//|   - LondonOpen requires all 3 TF agreement                       |
//|   - Breakeven stop after 1R move in favour                       |
//|   - Session-specific TP multipliers                               |
//|   - Hard block on RANGING regime                                  |
//|   - Daily loss limit and consecutive-loss circuit breaker         |
//+------------------------------------------------------------------+
#property copyright "XAUUSD-Backtest Project"
#property version   "1.02"
#property strict

#include <Trade\Trade.mqh>

//--- Risk sizing
input bool   InpUseRiskPercent = false;  // true = risk % of balance | false = fixed USD
input double InpRiskPercent    = 1.0;    // Risk % of account balance (if UseRiskPercent=true)
input double InpRiskUSD        = 100.0;  // Fixed risk per trade USD (if UseRiskPercent=false)

//--- Core indicator periods
input int    InpATRPeriod      = 14;     // ATR period (5M chart)
input int    InpEMAPeriod      = 50;     // EMA period for trend filter (15M)
input int    InpADXPeriod      = 14;     // ADX period (15M)

//--- ATR-based SL/TP
input double InpSLATRMult      = 1.5;   // SL = entry ± SLMult × ATR
input double InpBaseRR         = 2.5;   // Base reward/risk ratio

//--- ADX regime thresholds
input double InpADXTrend       = 25.0;  // ADX > this → TRENDING
input double InpADXRange       = 20.0;  // ADX < this → RANGING

//--- Signal quality
input double InpMinConfidence  = 0.72;  // Minimum confluence score to trade
input double InpPinRatio       = 2.0;   // Min wick/body ratio for pin bar
input double InpPinBodyPct     = 0.30;  // Max body/total_range for pin bar

//--- Regime guards
input bool   InpBlockRanging   = true;  // Hard-block all entries when ADX < ADXRange

//--- Daily risk controls
input int    InpMaxConsecLosses = 4;    // Pause rest of day after N consecutive SL hits
input double InpMaxDailyLossPC  = 3.0; // Stop trading today if equity falls this % intraday

//--- Session max trades per day
input int InpMaxAsia        = 8;
input int InpMaxLondonOpen  = 2;
input int InpMaxLondon      = 3;
input int InpMaxOverlap     = 3;

//--- Session cooldown minutes
input int InpCooldownAsia       = 15;
input int InpCooldownLondonOpen = 45;
input int InpCooldownLondon     = 30;
input int InpCooldownOverlap    = 30;

//--- Session TP multipliers (× ATR)
input double InpTPAsia        = 3.5;
input double InpTPLondonOpen  = 2.0;
input double InpTPLondon      = 2.5;
input double InpTPOverlap     = 3.0;

//--- Regime TP adjustment
input double InpRegimeTrendBonus   = 0.20;  // +20% TP when TRENDING
input double InpRegimeRangePenalty = 0.10;  // -10% TP when RANGING

//--- EA identity
input int InpMagicNumber = 202600;

//--- Global handles
int g_atr_h  = INVALID_HANDLE;
int g_ema_h  = INVALID_HANDLE;
int g_adx_h  = INVALID_HANDLE;

//--- Trade object
CTrade g_trade;

//--- Bar tracking
datetime g_last_bar = 0;

//--- Session daily trade counters (reset each new day)
int      g_cnt_asia       = 0;
int      g_cnt_londonopen = 0;
int      g_cnt_london     = 0;
int      g_cnt_overlap    = 0;
datetime g_last_reset_day = 0;

//--- Session cooldown timestamps
datetime g_last_trade_asia       = 0;
datetime g_last_trade_londonopen = 0;
datetime g_last_trade_london     = 0;
datetime g_last_trade_overlap    = 0;

//--- Daily risk tracking
int    g_consec_losses    = 0;    // consecutive SL hits today
double g_day_start_equity = 0.0; // equity at start of trading day

//--- Regime strings
#define REGIME_TRENDING      "TRENDING"
#define REGIME_RANGING       "RANGING"
#define REGIME_TRANSITIONING "TRANSITIONING"

//--- Session strings
#define SESSION_ASIA        "Asia"
#define SESSION_LONDONOPEN  "LondonOpen"
#define SESSION_LONDON      "London"
#define SESSION_OVERLAP     "Overlap"
#define SESSION_NY          "NY"
#define SESSION_ROLLOVER    "Rollover"
#define SESSION_BLOCKED     "Blocked"

//+------------------------------------------------------------------+
int OnInit()
{
    g_atr_h = iATR(_Symbol, PERIOD_M5,  InpATRPeriod);
    g_ema_h = iMA (_Symbol, PERIOD_M15, InpEMAPeriod, 0, MODE_EMA, PRICE_CLOSE);
    g_adx_h = iADX(_Symbol, PERIOD_M15, InpADXPeriod);

    if(g_atr_h == INVALID_HANDLE || g_ema_h == INVALID_HANDLE || g_adx_h == INVALID_HANDLE)
    {
        Print("ERROR: indicator handle creation failed");
        return INIT_FAILED;
    }

    g_trade.SetExpertMagicNumber(InpMagicNumber);
    g_trade.SetDeviationInPoints(20);
    g_trade.SetTypeFilling(ORDER_FILLING_IOC);

    g_day_start_equity = AccountInfoDouble(ACCOUNT_EQUITY);

    Print("XAUUSD PA EA v1.02 initialized | BlockRanging=", InpBlockRanging,
          " | MaxConsecLosses=", InpMaxConsecLosses,
          " | MaxDailyLoss=", InpMaxDailyLossPC, "%");
    return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    IndicatorRelease(g_atr_h);
    IndicatorRelease(g_ema_h);
    IndicatorRelease(g_adx_h);
}

//+------------------------------------------------------------------+
//| Trade transaction handler — tracks consecutive losses             |
//+------------------------------------------------------------------+
void OnTradeTransaction(const MqlTradeTransaction &trans,
                        const MqlTradeRequest      &request,
                        const MqlTradeResult       &result)
{
    if(trans.type != TRADE_TRANSACTION_DEAL_ADD) return;

    if(!HistoryDealSelect(trans.deal)) return;
    if(HistoryDealGetInteger(trans.deal, DEAL_MAGIC) != InpMagicNumber) return;
    if(HistoryDealGetString(trans.deal, DEAL_SYMBOL) != _Symbol) return;

    ENUM_DEAL_ENTRY entry_type = (ENUM_DEAL_ENTRY)HistoryDealGetInteger(trans.deal, DEAL_ENTRY);
    if(entry_type != DEAL_ENTRY_OUT) return;  // only count closing deals

    double profit = HistoryDealGetDouble(trans.deal, DEAL_PROFIT)
                  + HistoryDealGetDouble(trans.deal, DEAL_SWAP)
                  + HistoryDealGetDouble(trans.deal, DEAL_COMMISSION);

    if(profit < 0.0)
    {
        g_consec_losses++;
        Print("Consecutive losses: ", g_consec_losses, "/", InpMaxConsecLosses);
        if(g_consec_losses >= InpMaxConsecLosses)
            Print("Circuit breaker triggered — no new trades for rest of day");
    }
    else
    {
        g_consec_losses = 0;
    }
}

//+------------------------------------------------------------------+
void OnTick()
{
    //--- Manage breakeven on every tick for open positions
    ManageBreakeven();

    //--- Only act on new M5 bar
    datetime bar_time = iTime(_Symbol, PERIOD_M5, 0);
    if(bar_time == g_last_bar) return;
    g_last_bar = bar_time;

    //--- Reset daily counters on new day
    ResetDailyCounters();

    //--- Circuit breaker: consecutive losses
    if(g_consec_losses >= InpMaxConsecLosses) return;

    //--- Circuit breaker: daily drawdown
    if(GetDailyDrawdownPC() >= InpMaxDailyLossPC)
    {
        Print("Daily loss limit reached (", DoubleToString(GetDailyDrawdownPC(), 2),
              "%) — no new trades today");
        return;
    }

    //--- Get current UTC time
    MqlDateTime dt;
    TimeToStruct(TimeGMT(), dt);
    int h = dt.hour;

    //--- Classify session
    string session = GetSession(h);
    if(session == SESSION_NY || session == SESSION_ROLLOVER || session == SESSION_BLOCKED)
        return;

    //--- Check session trade cap
    if(!CanTradeSession(session)) return;

    //--- Check session cooldown
    if(!CooldownClear(session)) return;

    //--- Require at least 50 bars on M5 for indicator warmup
    if(iBars(_Symbol, PERIOD_M5) < 50) return;

    //--- Read indicators
    double atr_val = GetBuffer(g_atr_h, 1);   // Last closed M5 bar
    double ema_val = GetBuffer(g_ema_h, 1);   // Last closed M15 bar (aligned)
    double adx_val = GetBuffer(g_adx_h, 1);   // ADX main on M15

    if(atr_val <= 0 || ema_val <= 0) return;

    //--- Regime
    string regime = ClassifyRegime(adx_val);

    //--- Hard block on RANGING regime
    if(InpBlockRanging && regime == REGIME_RANGING) return;

    //--- Price-action votes per timeframe
    int    vote15m = 0, vote5m = 0, vote1m = 0;
    double qual15m = 0, qual5m = 0, qual1m = 0;
    string pat15m  = "", pat5m = "", pat1m = "";

    GetPAVote(PERIOD_M15, ema_val, vote15m, qual15m, pat15m);
    GetPAVote(PERIOD_M5,  ema_val, vote5m,  qual5m,  pat5m);
    GetPAVote(PERIOD_M1,  ema_val, vote1m,  qual1m,  pat1m);

    //--- Determine direction from votes
    int bull = (vote15m == 1 ? 1 : 0) + (vote5m == 1 ? 1 : 0) + (vote1m == 1 ? 1 : 0);
    int bear = (vote15m == -1 ? 1 : 0) + (vote5m == -1 ? 1 : 0) + (vote1m == -1 ? 1 : 0);

    int    agree    = 0;
    int    dir      = 0;   // 1 = BUY, -1 = SELL, 0 = no signal
    double avg_qual = 0;

    if(bull > bear && bull >= 1)
    {
        dir      = 1;
        agree    = bull;
        avg_qual = ((vote15m==1?qual15m:0) + (vote5m==1?qual5m:0) + (vote1m==1?qual1m:0)) / agree;
    }
    else if(bear > bull && bear >= 1)
    {
        dir      = -1;
        agree    = bear;
        avg_qual = ((vote15m==-1?qual15m:0) + (vote5m==-1?qual5m:0) + (vote1m==-1?qual1m:0)) / agree;
    }

    if(dir == 0) return;

    //--- LondonOpen strict: require all 3 TF agreement
    if(session == SESSION_LONDONOPEN && agree < 3) return;

    //--- Confidence score
    double confidence = MathMin(0.45 + 0.20 * agree, 1.0);
    confidence = MathMin(confidence + 0.10 * avg_qual, 1.0);
    if(regime == REGIME_TRANSITIONING) confidence *= 0.90;

    if(confidence < InpMinConfidence) return;

    //--- Max 3 positions open at once
    if(CountOpenPositions() >= 3) return;

    //--- Session TP multiplier
    double tp_mult = GetTPMultiplier(session);
    if(regime == REGIME_TRENDING) tp_mult *= (1.0 + InpRegimeTrendBonus);
    if(regime == REGIME_RANGING)  tp_mult *= (1.0 - InpRegimeRangePenalty);

    //--- SL / TP distances
    double sl_dist = InpSLATRMult * atr_val;
    double tp_dist = tp_mult      * atr_val;

    //--- Position size
    double risk_usd = InpUseRiskPercent
        ? AccountInfoDouble(ACCOUNT_BALANCE) * InpRiskPercent / 100.0
        : InpRiskUSD;
    double contract_size = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_CONTRACT_SIZE);
    if(contract_size <= 0) contract_size = 100.0;
    double sl_dist_usd = sl_dist * contract_size;
    if(sl_dist_usd <= 0) return;
    double lots = NormalizeDouble(risk_usd / sl_dist_usd, 2);
    double min_lot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
    double max_lot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
    lots = MathMax(MathMin(lots, max_lot), min_lot);

    //--- Entry price
    double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
    double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);

    if(dir == 1)
    {
        double entry = ask;
        double sl    = NormalizeDouble(entry - sl_dist, _Digits);
        double tp    = NormalizeDouble(entry + tp_dist, _Digits);
        if(g_trade.Buy(lots, _Symbol, entry, sl, tp, "PA_EA"))
            RecordSessionTrade(session);
    }
    else
    {
        double entry = bid;
        double sl    = NormalizeDouble(entry + sl_dist, _Digits);
        double tp    = NormalizeDouble(entry - tp_dist, _Digits);
        if(g_trade.Sell(lots, _Symbol, entry, sl, tp, "PA_EA"))
            RecordSessionTrade(session);
    }
}

//+------------------------------------------------------------------+
//| Returns intraday equity drawdown as a percentage                  |
//+------------------------------------------------------------------+
double GetDailyDrawdownPC()
{
    if(g_day_start_equity <= 0) return 0.0;
    double equity = AccountInfoDouble(ACCOUNT_EQUITY);
    double drop   = g_day_start_equity - equity;
    return (drop / g_day_start_equity) * 100.0;
}

//+------------------------------------------------------------------+
//| Classify session from UTC hour                                    |
//+------------------------------------------------------------------+
string GetSession(int h)
{
    if(h >= 0  && h < 7)  return SESSION_ASIA;
    if(h >= 7  && h < 9)  return SESSION_LONDONOPEN;
    if(h >= 9  && h < 12) return SESSION_LONDON;
    if(h >= 12 && h < 14) return SESSION_OVERLAP;
    if(h >= 14 && h < 17) return SESSION_NY;
    return SESSION_ROLLOVER;
}

//+------------------------------------------------------------------+
//| ADX-based regime classification                                   |
//+------------------------------------------------------------------+
string ClassifyRegime(double adx_val)
{
    if(adx_val > InpADXTrend) return REGIME_TRENDING;
    if(adx_val < InpADXRange) return REGIME_RANGING;
    return REGIME_TRANSITIONING;
}

//+------------------------------------------------------------------+
//| Detect PA pattern and return directional vote                     |
//| vote: 1=BUY, -1=SELL, 0=none   qual: 0.0-1.0                    |
//+------------------------------------------------------------------+
void GetPAVote(ENUM_TIMEFRAMES tf, double ema15m,
               int &vote, double &qual, string &pat)
{
    vote = 0; qual = 0.0; pat = "";

    if(iBars(_Symbol, tf) < 4) return;

    double o1 = iOpen (_Symbol, tf, 1);
    double h1 = iHigh (_Symbol, tf, 1);
    double l1 = iLow  (_Symbol, tf, 1);
    double c1 = iClose(_Symbol, tf, 1);

    double o2 = iOpen (_Symbol, tf, 2);
    double h2 = iHigh (_Symbol, tf, 2);
    double l2 = iLow  (_Symbol, tf, 2);
    double c2 = iClose(_Symbol, tf, 2);

    double dummy_q = 0;
    bool bull_pin1 = IsBullPinBar(o1, h1, l1, c1, dummy_q);
    bool bear_pin1 = IsBearPinBar(o1, h1, l1, c1, dummy_q);
    bool bull_pin2 = IsBullPinBar(o2, h2, l2, c2, dummy_q);
    bool bear_pin2 = IsBearPinBar(o2, h2, l2, c2, dummy_q);

    double q_eng  = 0;
    bool bull_eng = IsBullEngulfing(o2, h2, l2, c2, o1, h1, l1, c1, q_eng);
    bool bear_eng = IsBearEngulfing(o2, h2, l2, c2, o1, h1, l1, c1, q_eng);

    bool has_bull = bull_pin1 || bull_pin2 || bull_eng;
    bool has_bear = bear_pin1 || bear_pin2 || bear_eng;

    if(!has_bull && !has_bear) return;

    double best_q  = 0;
    string best_pat = "";
    int    raw_vote = 0;

    if(bull_pin1 || bull_pin2)
    {
        double q = 0;
        IsBullPinBar(bull_pin1 ? o1 : o2, bull_pin1 ? h1 : h2,
                     bull_pin1 ? l1 : l2, bull_pin1 ? c1 : c2, q);
        if(q > best_q) { best_q = q; raw_vote = 1; best_pat = "pin_bar"; }
    }
    if(bear_pin1 || bear_pin2)
    {
        double q = 0;
        IsBearPinBar(bear_pin1 ? o1 : o2, bear_pin1 ? h1 : h2,
                     bear_pin1 ? l1 : l2, bear_pin1 ? c1 : c2, q);
        if(q > best_q) { best_q = q; raw_vote = -1; best_pat = "pin_bar"; }
    }
    if(bull_eng && q_eng > best_q) { best_q = q_eng; raw_vote = 1;  best_pat = "engulfing"; }
    if(bear_eng && q_eng > best_q) { best_q = q_eng; raw_vote = -1; best_pat = "engulfing"; }

    if(raw_vote == 0) return;

    //--- EMA50 trend filter (only applied to M15 vote)
    if(tf == PERIOD_M15)
    {
        bool above_ema = (c1 > ema15m);
        if(raw_vote ==  1 && !above_ema) return;
        if(raw_vote == -1 &&  above_ema) return;
    }

    vote = raw_vote;
    qual = best_q;
    pat  = best_pat;
}

//+------------------------------------------------------------------+
//| Pin bar detection                                                 |
//+------------------------------------------------------------------+
bool IsBullPinBar(double o, double h, double l, double c, double &qual)
{
    double body       = MathAbs(c - o);
    double total      = h - l;
    if(total < 1e-9) return false;
    double lower_wick = MathMin(o, c) - l;
    if((body / total) > InpPinBodyPct) return false;
    if(lower_wick < InpPinRatio * MathMax(body, 0.001)) return false;
    qual = MathMin(lower_wick / total, 1.0);
    return true;
}

bool IsBearPinBar(double o, double h, double l, double c, double &qual)
{
    double body       = MathAbs(c - o);
    double total      = h - l;
    if(total < 1e-9) return false;
    double upper_wick = h - MathMax(o, c);
    if((body / total) > InpPinBodyPct) return false;
    if(upper_wick < InpPinRatio * MathMax(body, 0.001)) return false;
    qual = MathMin(upper_wick / total, 1.0);
    return true;
}

//+------------------------------------------------------------------+
//| Engulfing detection                                               |
//+------------------------------------------------------------------+
bool IsBullEngulfing(double o1, double h1, double l1, double c1,
                     double o2, double h2, double l2, double c2,
                     double &qual)
{
    if(c1 >= o1) return false;
    if(c2 <= o2) return false;
    if(c2 < MathMax(o1, c1)) return false;
    if(o2 > MathMin(o1, c1)) return false;
    double body1 = MathAbs(c1 - o1);
    double body2 = MathAbs(c2 - o2);
    if(body2 <= body1) return false;
    qual = MathMin(body2 / MathMax(body1, 0.001) - 1.0, 1.0);
    return true;
}

bool IsBearEngulfing(double o1, double h1, double l1, double c1,
                     double o2, double h2, double l2, double c2,
                     double &qual)
{
    if(c1 <= o1) return false;
    if(c2 >= o2) return false;
    if(c2 > MathMin(o1, c1)) return false;
    if(o2 < MathMax(o1, c1)) return false;
    double body1 = MathAbs(c1 - o1);
    double body2 = MathAbs(c2 - o2);
    if(body2 <= body1) return false;
    qual = MathMin(body2 / MathMax(body1, 0.001) - 1.0, 1.0);
    return true;
}

//+------------------------------------------------------------------+
//| Breakeven management — runs on every tick                         |
//+------------------------------------------------------------------+
void ManageBreakeven()
{
    for(int i = PositionsTotal() - 1; i >= 0; i--)
    {
        ulong ticket = PositionGetTicket(i);
        if(!PositionSelectByTicket(ticket)) continue;
        if(PositionGetInteger(POSITION_MAGIC) != InpMagicNumber) continue;

        double entry  = PositionGetDouble(POSITION_PRICE_OPEN);
        double cur_sl = PositionGetDouble(POSITION_SL);
        double cur_tp = PositionGetDouble(POSITION_TP);
        double cur_bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
        double cur_ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);

        ENUM_POSITION_TYPE ptype = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
        double sl_dist = MathAbs(cur_tp - entry) / InpBaseRR;

        if(ptype == POSITION_TYPE_BUY)
        {
            if(cur_bid >= entry + sl_dist && cur_sl < entry)
                g_trade.PositionModify(ticket, NormalizeDouble(entry + _Point, _Digits), cur_tp);
        }
        else
        {
            if(cur_ask <= entry - sl_dist && cur_sl > entry)
                g_trade.PositionModify(ticket, NormalizeDouble(entry - _Point, _Digits), cur_tp);
        }
    }
}

//+------------------------------------------------------------------+
//| Session helpers                                                   |
//+------------------------------------------------------------------+
bool CanTradeSession(const string &session)
{
    if(session == SESSION_ASIA       && g_cnt_asia       >= InpMaxAsia)       return false;
    if(session == SESSION_LONDONOPEN && g_cnt_londonopen >= InpMaxLondonOpen) return false;
    if(session == SESSION_LONDON     && g_cnt_london     >= InpMaxLondon)     return false;
    if(session == SESSION_OVERLAP    && g_cnt_overlap    >= InpMaxOverlap)    return false;
    return true;
}

bool CooldownClear(const string &session)
{
    datetime now = TimeGMT();
    if(session == SESSION_ASIA       && g_last_trade_asia       > 0 && (now - g_last_trade_asia)       < InpCooldownAsia       * 60) return false;
    if(session == SESSION_LONDONOPEN && g_last_trade_londonopen > 0 && (now - g_last_trade_londonopen) < InpCooldownLondonOpen * 60) return false;
    if(session == SESSION_LONDON     && g_last_trade_london     > 0 && (now - g_last_trade_london)     < InpCooldownLondon     * 60) return false;
    if(session == SESSION_OVERLAP    && g_last_trade_overlap    > 0 && (now - g_last_trade_overlap)    < InpCooldownOverlap    * 60) return false;
    return true;
}

void RecordSessionTrade(const string &session)
{
    datetime now = TimeGMT();
    if(session == SESSION_ASIA)       { g_cnt_asia++;       g_last_trade_asia       = now; }
    if(session == SESSION_LONDONOPEN) { g_cnt_londonopen++; g_last_trade_londonopen = now; }
    if(session == SESSION_LONDON)     { g_cnt_london++;     g_last_trade_london     = now; }
    if(session == SESSION_OVERLAP)    { g_cnt_overlap++;    g_last_trade_overlap    = now; }
}

void ResetDailyCounters()
{
    MqlDateTime today;
    TimeToStruct(TimeGMT(), today);
    datetime day_start = StructToTime(today) - today.hour * 3600 - today.min * 60 - today.sec;
    if(g_last_reset_day == day_start) return;
    g_last_reset_day = day_start;

    g_cnt_asia = g_cnt_londonopen = g_cnt_london = g_cnt_overlap = 0;
    g_last_trade_asia = g_last_trade_londonopen = g_last_trade_london = g_last_trade_overlap = 0;

    //--- Reset daily risk trackers
    g_consec_losses    = 0;
    g_day_start_equity = AccountInfoDouble(ACCOUNT_EQUITY);

    Print("Daily reset | equity=", DoubleToString(g_day_start_equity, 2));
}

double GetTPMultiplier(const string &session)
{
    if(session == SESSION_ASIA)       return InpTPAsia;
    if(session == SESSION_LONDONOPEN) return InpTPLondonOpen;
    if(session == SESSION_LONDON)     return InpTPLondon;
    if(session == SESSION_OVERLAP)    return InpTPOverlap;
    return InpBaseRR;
}

//+------------------------------------------------------------------+
//| Utility: read a buffer value safely                               |
//+------------------------------------------------------------------+
double GetBuffer(int handle, int shift)
{
    double buf[1];
    if(CopyBuffer(handle, 0, shift, 1, buf) != 1) return -1.0;
    return buf[0];
}

//+------------------------------------------------------------------+
//| Count open positions belonging to this EA                        |
//+------------------------------------------------------------------+
int CountOpenPositions()
{
    int count = 0;
    for(int i = 0; i < PositionsTotal(); i++)
    {
        ulong ticket = PositionGetTicket(i);
        if(PositionSelectByTicket(ticket) &&
           PositionGetInteger(POSITION_MAGIC) == InpMagicNumber &&
           PositionGetString(POSITION_SYMBOL) == _Symbol)
            count++;
    }
    return count;
}
//+------------------------------------------------------------------+
