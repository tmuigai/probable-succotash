
import { useState, useEffect } from 'react';
import { Link } from 'wouter';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { FlaskConical, CircleDot } from 'lucide-react';
import { useWebSocket } from '@/hooks/useWebSocket';
import { useMarketData, useICTAnalysis, useSignals, useTrades, useAccountMetrics } from '@/hooks/useMarketData';
import { Chart } from '@/components/trading/Chart';
import { OrderBlocks } from '@/components/trading/OrderBlocks';
import { MarketStructure } from '@/components/trading/MarketStructure';
import { Signals } from '@/components/trading/Signals';
import { RiskManagement } from '@/components/trading/RiskManagement';
import { KillZones } from '@/components/trading/KillZones';
import { Performance } from '@/components/trading/Performance';
import { PO3Analysis } from '@/components/trading/PO3Analysis';
import { LiquidityAnalysis } from '@/components/trading/LiquidityAnalysis';
import { EnhancedKillZones } from '@/components/trading/EnhancedKillZones';

export default function Dashboard() {
  const [selectedSymbol, setSelectedSymbol] = useState('EURUSD');
  const [selectedTimeframe, setSelectedTimeframe] = useState('M15');
  const [showOrderBlocks, setShowOrderBlocks] = useState(true);
  const [showFVG, setShowFVG] = useState(false);
  const [showStructure, setShowStructure] = useState(true);

  // WebSocket connection for real-time data
  const { isConnected, data: wsData, latestPrice } = useWebSocket();

  // API data hooks
  const { data: marketData = [], isLoading: loadingMarketData } = useMarketData(selectedSymbol, selectedTimeframe);
  const { data: ictAnalysis, isLoading: loadingAnalysis } = useICTAnalysis(selectedSymbol, selectedTimeframe);
  const { data: signals = [] } = useSignals();
  const { data: trades = [] } = useTrades();
  const { data: accountMetrics } = useAccountMetrics();

  // Get current prices for all symbols
  const currentPrices: { [symbol: string]: number } = {};
  wsData.currentPrices.forEach(price => {
    currentPrices[price.symbol] = price.bid;
  });

  const currentPrice = currentPrices[selectedSymbol] || 0;
  const priceChange = latestPrice?.symbol === selectedSymbol ? 
    (latestPrice.bid - (marketData[marketData.length - 1] ? parseFloat(marketData[marketData.length - 1].close) : latestPrice.bid)) : 0;
  const priceChangePercent = marketData.length > 0 ? 
    (priceChange / parseFloat(marketData[marketData.length - 1].close)) * 100 : 0;

  return (
    <div className="min-h-screen bg-gray-900 text-gray-100">
      {/* Header */}
      <header className="bg-gray-800 border-b border-gray-700 px-4 py-3">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
          <div className="flex flex-col lg:flex-row lg:items-center gap-4">
            <h1 className="text-xl font-bold text-white">ICT Smart Money Dashboard</h1>
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <CircleDot className={`w-2 h-2 ${wsData.marketStatus ? 'text-green-400' : 'text-red-400'} ${wsData.marketStatus ? 'animate-pulse' : ''}`} />
                <span className="text-sm text-gray-300">
                  {wsData.marketStatus ? 'Market Open' : 'Market Closed'}
                </span>
              </div>
              <div className="text-sm text-gray-300">
                <span>{wsData.currentSession}</span>
              </div>
            </div>
          </div>
          
          <div className="flex flex-col lg:flex-row lg:items-center gap-4">
            <div className="flex items-center gap-3">
              <Select value={selectedSymbol} onValueChange={setSelectedSymbol}>
                <SelectTrigger className="w-32 bg-gray-700 border-gray-600">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent className="bg-gray-700 border-gray-600">
                  <SelectItem value="EURUSD">EURUSD</SelectItem>
                  <SelectItem value="GBPUSD">GBPUSD</SelectItem>
                  <SelectItem value="USDJPY">USDJPY</SelectItem>
                  <SelectItem value="USDCHF">USDCHF</SelectItem>
                  <SelectItem value="AUDUSD">AUDUSD</SelectItem>
                </SelectContent>
              </Select>
              
              <Link href="/performance">
                <Button variant="outline" className="bg-blue-600 border-blue-500 hover:bg-blue-700 text-white">
                  Performance Dashboard
                </Button>
              </Link>
              
              <Button className="bg-blue-600 hover:bg-blue-700 px-4 py-2 text-sm font-medium">
                <FlaskConical className="w-4 h-4 mr-2" />
                Paper Trading
              </Button>
            </div>
            
            {accountMetrics && (
              <div className="flex flex-wrap items-center gap-4 text-sm">
                <div>
                  <span className="text-gray-400">Balance:</span>
                  <span className="text-white font-semibold ml-1">
                    ${accountMetrics.balance.toFixed(2)}
                  </span>
                </div>
                <div>
                  <span className="text-gray-400">Equity:</span>
                  <span className="text-green-400 font-semibold ml-1">
                    ${accountMetrics.equity.toFixed(2)}
                  </span>
                </div>
                <div>
                  <span className="text-gray-400">P&L:</span>
                  <span className={`font-semibold ml-1 ${accountMetrics.dailyPnL >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                    {accountMetrics.dailyPnL >= 0 ? '+' : ''}${accountMetrics.dailyPnL.toFixed(2)}
                  </span>
                </div>
              </div>
            )}
          </div>
        </div>
      </header>

      <div className="flex flex-col lg:flex-row h-[calc(100vh-80px)]">
        {/* Left Sidebar */}
        <aside className="w-full lg:w-80 bg-gray-800 border-b lg:border-b-0 lg:border-r border-gray-700 overflow-y-auto">
          <div className="p-4 space-y-4">
            <EnhancedKillZones />
            <PO3Analysis symbol={selectedSymbol} />
            <LiquidityAnalysis symbol={selectedSymbol} />
            <KillZones killZones={wsData.killZones} />
            
            {ictAnalysis && (
              <MarketStructure marketStructure={ictAnalysis.marketStructure} />
            )}
            
            {ictAnalysis && (
              <OrderBlocks 
                orderBlocks={ictAnalysis.orderBlocks} 
                symbol={selectedSymbol} 
              />
            )}
          </div>
        </aside>

        {/* Main Content */}
        <main className="flex-1 flex flex-col min-h-0">
          {/* Chart Toolbar */}
          <div className="bg-gray-800 border-b border-gray-700 px-4 py-2">
            <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
              <div className="flex flex-col lg:flex-row lg:items-center gap-4">
                <div className="flex items-center space-x-2">
                  {['M1', 'M5', 'M15', 'H1', 'D1'].map((tf) => (
                    <Button
                      key={tf}
                      variant={selectedTimeframe === tf ? "default" : "ghost"}
                      size="sm"
                      onClick={() => setSelectedTimeframe(tf)}
                      className={`px-3 py-1 text-xs ${
                        selectedTimeframe === tf 
                          ? 'bg-blue-600 text-white' 
                          : 'text-gray-300 hover:text-white hover:bg-gray-700'
                      }`}
                    >
                      {tf}
                    </Button>
                  ))}
                </div>
                
                <div className="h-4 w-px bg-gray-600 hidden lg:block"></div>
                
                <div className="flex items-center space-x-2">
                  <Button
                    variant={showOrderBlocks ? "default" : "ghost"}
                    size="sm"
                    onClick={() => setShowOrderBlocks(!showOrderBlocks)}
                    className={`px-3 py-1 text-xs ${
                      showOrderBlocks 
                        ? 'bg-gray-600 text-white' 
                        : 'text-gray-300 hover:bg-gray-700'
                    }`}
                  >
                    Order Blocks
                  </Button>
                  <Button
                    variant={showFVG ? "default" : "ghost"}
                    size="sm"
                    onClick={() => setShowFVG(!showFVG)}
                    className={`px-3 py-1 text-xs ${
                      showFVG 
                        ? 'bg-gray-600 text-white' 
                        : 'text-gray-300 hover:bg-gray-700'
                    }`}
                  >
                    FVG
                  </Button>
                  <Button
                    variant={showStructure ? "default" : "ghost"}
                    size="sm"
                    onClick={() => setShowStructure(!showStructure)}
                    className={`px-3 py-1 text-xs ${
                      showStructure 
                        ? 'bg-gray-600 text-white' 
                        : 'text-gray-300 hover:bg-gray-700'
                    }`}
                  >
                    Structure
                  </Button>
                </div>
              </div>
              
              <div className="flex flex-wrap items-center gap-3 text-sm">
                <div>
                  <span className="text-gray-400">Current:</span>
                  <span className="text-white font-mono ml-1">
                    {currentPrice.toFixed(5)}
                  </span>
                </div>
                <div className={priceChange >= 0 ? 'text-green-400' : 'text-red-400'}>
                  <span>{priceChange >= 0 ? '+' : ''}{priceChange.toFixed(5)}</span>
                  <span className="ml-1">
                    ({priceChangePercent >= 0 ? '+' : ''}{priceChangePercent.toFixed(2)}%)
                  </span>
                </div>
                <div className="flex items-center space-x-1">
                  <CircleDot className={`w-2 h-2 ${isConnected ? 'text-green-400 animate-pulse' : 'text-red-400'}`} />
                  <span className="text-xs text-gray-400">
                    {isConnected ? 'Live' : 'Disconnected'}
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* Charts */}
          <div className="flex-1 p-4 min-h-0">
            <div className="grid grid-cols-1 xl:grid-cols-3 gap-4 h-full">
              {/* Main Chart */}
              <div className="xl:col-span-2 bg-gray-800 rounded-lg p-4 min-h-0">
                <div className="flex items-center justify-between mb-3">
                  <h3 className="text-sm font-semibold text-gray-200">
                    {selectedSymbol} {selectedTimeframe}
                  </h3>
                  <div className="flex items-center space-x-2 text-xs text-gray-400">
                    <span>Last update: {new Date().toLocaleTimeString()}</span>
                    <CircleDot className={`w-2 h-2 ${isConnected ? 'text-green-400 animate-pulse' : 'text-red-400'}`} />
                  </div>
                </div>
                
                {loadingMarketData ? (
                  <div className="w-full h-64 lg:h-96 bg-gray-900 rounded flex items-center justify-center">
                    <div className="text-gray-400">Loading chart data...</div>
                  </div>
                ) : (
                  <div className="h-64 lg:h-96">
                    <Chart
                      data={marketData}
                      orderBlocks={showOrderBlocks ? (ictAnalysis?.orderBlocks || []) : []}
                      fvgs={showFVG ? (ictAnalysis?.fvgs || []) : []}
                      title={`${selectedSymbol} ${selectedTimeframe}`}
                      height={400}
                      showOverlays={showOrderBlocks || showFVG || showStructure}
                    />
                  </div>
                )}
              </div>

              {/* Secondary Charts */}
              <div className="space-y-4">
                {/* M1 Internal Structure */}
                <div className="bg-gray-800 rounded-lg p-3">
                  <h4 className="text-xs font-semibold text-gray-300 mb-2">
                    M1 Internal Structure
                  </h4>
                  <div className="w-full h-32 bg-gray-900 rounded flex items-center justify-center">
                    <div className="text-gray-500 text-xs">M1 Chart</div>
                  </div>
                </div>

                {/* HTF Context */}
                <div className="bg-gray-800 rounded-lg p-3">
                  <h4 className="text-xs font-semibold text-gray-300 mb-2">
                    D1 HTF Context
                  </h4>
                  <div className="w-full h-32 bg-gray-900 rounded flex items-center justify-center">
                    <div className="text-gray-500 text-xs">D1 Chart</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </main>

        {/* Right Sidebar */}
        <aside className="w-full lg:w-80 bg-gray-800 border-t lg:border-t-0 lg:border-l border-gray-700 overflow-y-auto">
          <div className="p-4 space-y-4">
            <Signals signals={signals} />
            
            {accountMetrics ? (
              <RiskManagement accountMetrics={accountMetrics} />
            ) : (
              <div className="text-center py-4 text-gray-400 text-sm">
                Loading account metrics...
              </div>
            )}
            
            <Performance 
              trades={trades} 
              accountMetrics={accountMetrics || {
                balance: 10000,
                equity: 10000,
                margin: 0,
                freeMargin: 10000,
                marginLevel: 0,
                dailyPnL: 0,
                drawdown: 0,
                usedRiskToday: 0
              }} 
              currentPrices={currentPrices}
            />
          </div>
        </aside>
      </div>
    </div>
  );
}
