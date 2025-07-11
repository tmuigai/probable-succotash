import { useState, useEffect } from 'react';
import { Link } from 'wouter';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { LineChart, Line, AreaChart, Area, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';
import { TrendingUp, TrendingDown, DollarSign, Target, Activity, Clock, Award, AlertTriangle } from 'lucide-react';
import { useQuery } from '@tanstack/react-query';

interface PerformanceMetrics {
  totalTrades: number;
  winRate: number;
  profitFactor: number;
  totalPnL: number;
  averageWin: number;
  averageLoss: number;
  maxDrawdown: number;
  currentDrawdown: number;
  sharpeRatio: number;
  accountBalance: number;
  accountEquity: number;
  riskRewardRatio: number;
  consecutiveWins: number;
  consecutiveLosses: number;
  dailyPnL: number;
  weeklyPnL: number;
  monthlyPnL: number;
  bestTrade: number;
  worstTrade: number;
  tradingDays: number;
  averageTradesPerDay: number;
}

interface TradeAnalytics {
  equityCurve: Array<{ date: string; equity: number; drawdown: number }>;
  monthlyReturns: Array<{ month: string; return: number; trades: number }>;
  winLossDistribution: Array<{ range: string; wins: number; losses: number }>;
  tradingHours: Array<{ hour: number; pnl: number; trades: number }>;
  symbolPerformance: Array<{ symbol: string; trades: number; pnl: number; winRate: number }>;
  entryQuality: Array<{ setup: string; trades: number; winRate: number; avgPnL: number }>;
}

interface RiskMetrics {
  positionSizing: Array<{ size: string; percentage: number }>;
  riskPerTrade: number;
  maxRiskPerTrade: number;
  correlationRisk: number;
  portfolioHeatmap: Array<{ symbol: string; exposure: number; risk: number }>;
}

export function PerformanceDashboard() {
  const [timeframe, setTimeframe] = useState('1M');
  const [selectedMetric, setSelectedMetric] = useState('equity');

  const { data: metrics } = useQuery<PerformanceMetrics>({
    queryKey: ['/api/performance-metrics', timeframe],
    refetchInterval: 5000
  });

  const { data: analytics } = useQuery<TradeAnalytics>({
    queryKey: ['/api/trade-analytics', timeframe],
    refetchInterval: 10000
  });

  const { data: riskData } = useQuery<RiskMetrics>({
    queryKey: ['/api/risk-metrics'],
    refetchInterval: 15000
  });

  if (!metrics || !analytics || !riskData) {
    return (
      <div className="p-6 space-y-6">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          {[...Array(8)].map((_, i) => (
            <Card key={i} className="bg-gray-800 border-gray-700">
              <CardContent className="p-4">
                <div className="animate-pulse">
                  <div className="h-4 bg-gray-700 rounded w-3/4 mb-2"></div>
                  <div className="h-6 bg-gray-700 rounded w-1/2"></div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    );
  }

  const getMetricColor = (value: number, isPositive: boolean = true) => {
    if (isPositive) {
      return value >= 0 ? 'text-green-400' : 'text-red-400';
    }
    return value <= 0 ? 'text-green-400' : 'text-red-400';
  };

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2
    }).format(value);
  };

  const formatPercentage = (value: number) => {
    return `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`;
  };

  const COLORS = ['#10b981', '#f59e0b', '#ef4444', '#3b82f6', '#8b5cf6'];

  return (
    <div className="p-6 space-y-6 bg-gray-900 min-h-screen">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-white">Trading Performance Dashboard</h1>
          <p className="text-gray-400 mt-1">Real-time analytics and performance metrics</p>
        </div>
        <div className="flex items-center space-x-4">
          <Select value={timeframe} onValueChange={setTimeframe}>
            <SelectTrigger className="w-32 bg-gray-800 border-gray-700 text-white">
              <SelectValue />
            </SelectTrigger>
            <SelectContent className="bg-gray-800 border-gray-700">
              <SelectItem value="1D">1 Day</SelectItem>
              <SelectItem value="1W">1 Week</SelectItem>
              <SelectItem value="1M">1 Month</SelectItem>
              <SelectItem value="3M">3 Months</SelectItem>
              <SelectItem value="1Y">1 Year</SelectItem>
              <SelectItem value="ALL">All Time</SelectItem>
            </SelectContent>
          </Select>
          <Link href="/">
            <Button variant="outline" className="bg-gray-700 border-gray-600 hover:bg-gray-600 text-white">
              Trading Dashboard
            </Button>
          </Link>
          <Button className="bg-blue-600 hover:bg-blue-700">
            Export Report
          </Button>
        </div>
      </div>

      {/* Key Performance Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card className="bg-gray-800 border-gray-700">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-400 text-sm">Account Balance</p>
                <p className="text-2xl font-bold text-white">{formatCurrency(metrics.accountBalance)}</p>
                <p className={`text-sm ${getMetricColor(metrics.dailyPnL)}`}>
                  {formatPercentage(metrics.dailyPnL / metrics.accountBalance * 100)} today
                </p>
              </div>
              <DollarSign className="h-8 w-8 text-blue-400" />
            </div>
          </CardContent>
        </Card>

        <Card className="bg-gray-800 border-gray-700">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-400 text-sm">Total P&L</p>
                <p className={`text-2xl font-bold ${getMetricColor(metrics.totalPnL)}`}>
                  {formatCurrency(metrics.totalPnL)}
                </p>
                <p className="text-sm text-gray-400">{metrics.totalTrades} trades</p>
              </div>
              {metrics.totalPnL >= 0 ? 
                <TrendingUp className="h-8 w-8 text-green-400" /> : 
                <TrendingDown className="h-8 w-8 text-red-400" />
              }
            </div>
          </CardContent>
        </Card>

        <Card className="bg-gray-800 border-gray-700">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-400 text-sm">Win Rate</p>
                <p className="text-2xl font-bold text-white">{metrics.winRate.toFixed(1)}%</p>
                <Progress value={metrics.winRate} className="mt-2 h-2" />
              </div>
              <Target className="h-8 w-8 text-green-400" />
            </div>
          </CardContent>
        </Card>

        <Card className="bg-gray-800 border-gray-700">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-400 text-sm">Profit Factor</p>
                <p className={`text-2xl font-bold ${metrics.profitFactor >= 1.5 ? 'text-green-400' : metrics.profitFactor >= 1.0 ? 'text-yellow-400' : 'text-red-400'}`}>
                  {metrics.profitFactor.toFixed(2)}
                </p>
                <p className="text-sm text-gray-400">
                  {metrics.profitFactor >= 1.5 ? 'Excellent' : metrics.profitFactor >= 1.0 ? 'Good' : 'Poor'}
                </p>
              </div>
              <Award className="h-8 w-8 text-yellow-400" />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Secondary Metrics */}
      <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
        <Card className="bg-gray-800 border-gray-700">
          <CardContent className="p-3">
            <p className="text-gray-400 text-xs">Max Drawdown</p>
            <p className={`text-lg font-bold ${getMetricColor(metrics.maxDrawdown, false)}`}>
              {formatPercentage(metrics.maxDrawdown)}
            </p>
          </CardContent>
        </Card>

        <Card className="bg-gray-800 border-gray-700">
          <CardContent className="p-3">
            <p className="text-gray-400 text-xs">Sharpe Ratio</p>
            <p className={`text-lg font-bold ${metrics.sharpeRatio >= 1.5 ? 'text-green-400' : metrics.sharpeRatio >= 1.0 ? 'text-yellow-400' : 'text-red-400'}`}>
              {metrics.sharpeRatio.toFixed(2)}
            </p>
          </CardContent>
        </Card>

        <Card className="bg-gray-800 border-gray-700">
          <CardContent className="p-3">
            <p className="text-gray-400 text-xs">Risk/Reward</p>
            <p className="text-lg font-bold text-white">{metrics.riskRewardRatio.toFixed(2)}</p>
          </CardContent>
        </Card>

        <Card className="bg-gray-800 border-gray-700">
          <CardContent className="p-3">
            <p className="text-gray-400 text-xs">Avg Win</p>
            <p className="text-lg font-bold text-green-400">{formatCurrency(metrics.averageWin)}</p>
          </CardContent>
        </Card>

        <Card className="bg-gray-800 border-gray-700">
          <CardContent className="p-3">
            <p className="text-gray-400 text-xs">Avg Loss</p>
            <p className="text-lg font-bold text-red-400">{formatCurrency(metrics.averageLoss)}</p>
          </CardContent>
        </Card>

        <Card className="bg-gray-800 border-gray-700">
          <CardContent className="p-3">
            <p className="text-gray-400 text-xs">Best Trade</p>
            <p className="text-lg font-bold text-green-400">{formatCurrency(metrics.bestTrade)}</p>
          </CardContent>
        </Card>
      </div>

      {/* Main Analytics */}
      <Tabs defaultValue="equity" className="space-y-4">
        <TabsList className="bg-gray-800 border-gray-700">
          <TabsTrigger value="equity" className="data-[state=active]:bg-gray-700">Equity Curve</TabsTrigger>
          <TabsTrigger value="analysis" className="data-[state=active]:bg-gray-700">Trade Analysis</TabsTrigger>
          <TabsTrigger value="risk" className="data-[state=active]:bg-gray-700">Risk Management</TabsTrigger>
          <TabsTrigger value="patterns" className="data-[state=active]:bg-gray-700">ICT Patterns</TabsTrigger>
        </TabsList>

        <TabsContent value="equity" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <Card className="bg-gray-800 border-gray-700">
              <CardHeader>
                <CardTitle className="text-white">Equity Curve</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <AreaChart data={analytics.equityCurve}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis dataKey="date" stroke="#9ca3af" />
                    <YAxis stroke="#9ca3af" />
                    <Tooltip 
                      contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151' }}
                      labelStyle={{ color: '#f3f4f6' }}
                    />
                    <Area type="monotone" dataKey="equity" stroke="#10b981" fill="#10b981" fillOpacity={0.2} />
                  </AreaChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            <Card className="bg-gray-800 border-gray-700">
              <CardHeader>
                <CardTitle className="text-white">Drawdown Analysis</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <AreaChart data={analytics.equityCurve}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis dataKey="date" stroke="#9ca3af" />
                    <YAxis stroke="#9ca3af" />
                    <Tooltip 
                      contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151' }}
                      labelStyle={{ color: '#f3f4f6' }}
                    />
                    <Area type="monotone" dataKey="drawdown" stroke="#ef4444" fill="#ef4444" fillOpacity={0.3} />
                  </AreaChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </div>

          <Card className="bg-gray-800 border-gray-700">
            <CardHeader>
              <CardTitle className="text-white">Monthly Returns</CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={analytics.monthlyReturns}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis dataKey="month" stroke="#9ca3af" />
                  <YAxis stroke="#9ca3af" />
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151' }}
                    labelStyle={{ color: '#f3f4f6' }}
                  />
                  <Bar dataKey="return" fill="#3b82f6" />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="analysis" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <Card className="bg-gray-800 border-gray-700">
              <CardHeader>
                <CardTitle className="text-white">Win/Loss Distribution</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={analytics.winLossDistribution}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis dataKey="range" stroke="#9ca3af" />
                    <YAxis stroke="#9ca3af" />
                    <Tooltip 
                      contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151' }}
                      labelStyle={{ color: '#f3f4f6' }}
                    />
                    <Bar dataKey="wins" fill="#10b981" />
                    <Bar dataKey="losses" fill="#ef4444" />
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            <Card className="bg-gray-800 border-gray-700">
              <CardHeader>
                <CardTitle className="text-white">Trading Hours Performance</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={analytics.tradingHours}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis dataKey="hour" stroke="#9ca3af" />
                    <YAxis stroke="#9ca3af" />
                    <Tooltip 
                      contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151' }}
                      labelStyle={{ color: '#f3f4f6' }}
                    />
                    <Line type="monotone" dataKey="pnl" stroke="#10b981" strokeWidth={2} />
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </div>

          <Card className="bg-gray-800 border-gray-700">
            <CardHeader>
              <CardTitle className="text-white">Symbol Performance</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-gray-700">
                      <th className="text-left p-2 text-gray-400">Symbol</th>
                      <th className="text-center p-2 text-gray-400">Trades</th>
                      <th className="text-center p-2 text-gray-400">P&L</th>
                      <th className="text-center p-2 text-gray-400">Win Rate</th>
                      <th className="text-center p-2 text-gray-400">Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {analytics.symbolPerformance.map((symbol, index) => (
                      <tr key={index} className="border-b border-gray-700/50">
                        <td className="p-2 text-white font-medium">{symbol.symbol}</td>
                        <td className="p-2 text-center text-gray-300">{symbol.trades}</td>
                        <td className={`p-2 text-center font-medium ${getMetricColor(symbol.pnl)}`}>
                          {formatCurrency(symbol.pnl)}
                        </td>
                        <td className="p-2 text-center text-gray-300">{symbol.winRate.toFixed(1)}%</td>
                        <td className="p-2 text-center">
                          <Badge variant={symbol.pnl >= 0 ? "default" : "destructive"} className="text-xs">
                            {symbol.pnl >= 0 ? 'Profitable' : 'Loss'}
                          </Badge>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="risk" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <Card className="bg-gray-800 border-gray-700">
              <CardHeader>
                <CardTitle className="text-white">Position Sizing Distribution</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <PieChart>
                    <Pie
                      data={riskData.positionSizing}
                      cx="50%"
                      cy="50%"
                      outerRadius={80}
                      fill="#8884d8"
                      dataKey="percentage"
                      label={({ size, percentage }) => `${size}: ${percentage}%`}
                    >
                      {riskData.positionSizing.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            <Card className="bg-gray-800 border-gray-700">
              <CardHeader>
                <CardTitle className="text-white">Risk Metrics</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex justify-between items-center">
                  <span className="text-gray-400">Risk per Trade</span>
                  <span className="text-white font-medium">{riskData.riskPerTrade.toFixed(2)}%</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-400">Max Risk per Trade</span>
                  <span className="text-white font-medium">{riskData.maxRiskPerTrade.toFixed(2)}%</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-400">Correlation Risk</span>
                  <span className={`font-medium ${riskData.correlationRisk > 0.7 ? 'text-red-400' : riskData.correlationRisk > 0.4 ? 'text-yellow-400' : 'text-green-400'}`}>
                    {(riskData.correlationRisk * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-400">Current Drawdown</span>
                  <span className={`font-medium ${getMetricColor(metrics.currentDrawdown, false)}`}>
                    {formatPercentage(metrics.currentDrawdown)}
                  </span>
                </div>
              </CardContent>
            </Card>
          </div>

          <Card className="bg-gray-800 border-gray-700">
            <CardHeader>
              <CardTitle className="text-white">Portfolio Heat Map</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
                {riskData.portfolioHeatmap.map((item, index) => (
                  <div key={index} className="text-center">
                    <div 
                      className={`w-full h-20 rounded-lg flex items-center justify-center text-white font-bold ${
                        item.risk > 0.7 ? 'bg-red-600' : 
                        item.risk > 0.4 ? 'bg-yellow-600' : 
                        'bg-green-600'
                      }`}
                    >
                      {item.symbol}
                    </div>
                    <p className="text-xs text-gray-400 mt-1">{item.exposure.toFixed(1)}% exposure</p>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="patterns" className="space-y-4">
          <Card className="bg-gray-800 border-gray-700">
            <CardHeader>
              <CardTitle className="text-white">ICT Entry Quality Analysis</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-gray-700">
                      <th className="text-left p-2 text-gray-400">Setup Type</th>
                      <th className="text-center p-2 text-gray-400">Trades</th>
                      <th className="text-center p-2 text-gray-400">Win Rate</th>
                      <th className="text-center p-2 text-gray-400">Avg P&L</th>
                      <th className="text-center p-2 text-gray-400">Quality</th>
                    </tr>
                  </thead>
                  <tbody>
                    {analytics.entryQuality.map((entry, index) => (
                      <tr key={index} className="border-b border-gray-700/50">
                        <td className="p-2 text-white font-medium">{entry.setup}</td>
                        <td className="p-2 text-center text-gray-300">{entry.trades}</td>
                        <td className="p-2 text-center text-gray-300">{entry.winRate.toFixed(1)}%</td>
                        <td className={`p-2 text-center font-medium ${getMetricColor(entry.avgPnL)}`}>
                          {formatCurrency(entry.avgPnL)}
                        </td>
                        <td className="p-2 text-center">
                          <Badge 
                            variant={entry.winRate >= 70 ? "default" : entry.winRate >= 50 ? "secondary" : "destructive"} 
                            className="text-xs"
                          >
                            {entry.winRate >= 70 ? 'Excellent' : entry.winRate >= 50 ? 'Good' : 'Poor'}
                          </Badge>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}