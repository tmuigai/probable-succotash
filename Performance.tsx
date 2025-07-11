import { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { TrendingUp, TrendingDown, DollarSign, Clock, Target } from 'lucide-react';

interface Trade {
  id: string;
  symbol: string;
  type: 'buy' | 'sell';
  size: number;
  entryPrice: number;
  exitPrice?: number;
  pnl: string;
  status: 'open' | 'closed';
  timestamp: string;
}

interface AccountMetrics {
  balance: number;
  equity: number;
  margin: number;
  freeMargin: number;
  marginLevel: number;
  dailyPnL: number;
  drawdown: number;
  usedRiskToday?: number;
}

interface PerformanceProps {
  trades: Trade[];
  accountMetrics: AccountMetrics;
  currentPrices: { [symbol: string]: number };
}

export function Performance({ trades, accountMetrics, currentPrices }: PerformanceProps) {
  const [selectedPeriod, setSelectedPeriod] = useState('today');

  const openTrades = trades.filter(trade => trade.status === 'open');
  const closedTrades = trades.filter(trade => trade.status === 'closed');

  const totalPnL = closedTrades.reduce((sum, trade) => sum + parseFloat(trade.pnl), 0);
  const winningTrades = closedTrades.filter(trade => parseFloat(trade.pnl) > 0);
  const winRate = closedTrades.length > 0 ? (winningTrades.length / closedTrades.length) * 100 : 0;

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2
    }).format(value);
  };

  const getPnLColor = (pnl: number) => {
    return pnl >= 0 ? 'text-green-400' : 'text-red-400';
  };

  return (
    <div className="space-y-4">
      <Card className="bg-gray-800 border-gray-700">
        <CardHeader className="pb-3">
          <CardTitle className="text-sm font-semibold text-gray-200 flex items-center">
            <Target className="w-4 h-4 mr-2 text-blue-400" />
            Performance
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="flex justify-between items-center">
            <span className="text-sm text-gray-300">Today's P&L</span>
            <span className={`text-sm font-medium ${getPnLColor(accountMetrics?.dailyPnL || 0)}`}>
              {formatCurrency(accountMetrics?.dailyPnL || 0)}
            </span>
          </div>

          <div className="flex justify-between items-center">
            <span className="text-sm text-gray-300">Win Rate</span>
            <span className="text-white text-sm font-medium">{winRate.toFixed(1)}%</span>
          </div>

          <div className="flex justify-between items-center">
            <span className="text-sm text-gray-300">Total Trades</span>
            <span className="text-white text-sm font-medium">{closedTrades.length}</span>
          </div>

          <div className="flex justify-between items-center">
            <span className="text-sm text-gray-300">Open Positions</span>
            <span className="text-white text-sm font-medium">{openTrades.length}</span>
          </div>

          <div className="flex justify-between items-center">
            <span className="text-sm text-gray-300">Account Equity</span>
            <span className="text-white text-sm font-medium">
              {formatCurrency(accountMetrics?.equity || 0)}
            </span>
          </div>
        </CardContent>
      </Card>

      {/* Recent Trades */}
      <Card className="bg-gray-800 border-gray-700">
        <CardHeader className="pb-3">
          <CardTitle className="text-sm font-semibold text-gray-200 flex items-center">
            <Clock className="w-4 h-4 mr-2 text-yellow-400" />
            Recent Trades
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-2">
            {trades.slice(0, 5).map((trade) => (
              <div key={trade.id} className="flex items-center justify-between p-2 bg-gray-700 rounded">
                <div className="flex items-center space-x-2">
                  <Badge variant={trade.type === 'buy' ? 'default' : 'destructive'} className="text-xs">
                    {trade.type.toUpperCase()}
                  </Badge>
                  <span className="text-sm text-white">{trade.symbol}</span>
                </div>
                <div className="flex items-center space-x-2">
                  <span className={`text-sm font-medium ${getPnLColor(parseFloat(trade.pnl))}`}>
                    {formatCurrency(parseFloat(trade.pnl))}
                  </span>
                  {parseFloat(trade.pnl) >= 0 ? 
                    <TrendingUp className="w-3 h-3 text-green-400" /> : 
                    <TrendingDown className="w-3 h-3 text-red-400" />
                  }
                </div>
              </div>
            ))}
            {trades.length === 0 && (
              <div className="text-center py-4 text-gray-400 text-sm">
                No trades yet
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Quick Actions */}
      <Card className="bg-gray-800 border-gray-700">
        <CardHeader className="pb-3">
          <CardTitle className="text-sm font-semibold text-gray-200">Quick Actions</CardTitle>
        </CardHeader>
        <CardContent className="space-y-2">
          <Button variant="outline" size="sm" className="w-full text-xs">
            View Full Report
          </Button>
          <Button variant="outline" size="sm" className="w-full text-xs">
            Export Trades
          </Button>
        </CardContent>
      </Card>
    </div>
  );
}