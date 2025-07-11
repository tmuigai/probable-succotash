import { useQuery } from '@tanstack/react-query';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { AlertTriangle, Target, Droplets } from 'lucide-react';

interface LiquidityPool {
  id: string;
  type: 'buy_stops' | 'sell_stops' | 'equal_highs' | 'equal_lows';
  price: number;
  strength: number;
  timestamp: string;
  tested: boolean;
  swept: boolean;
}

interface LiquidityAnalysis {
  pools: LiquidityPool[];
  recentSweeps: LiquidityPool[];
  nextLikelyTarget: LiquidityPool | null;
  liquiditySide: 'buy_side' | 'sell_side' | 'balanced';
}

interface LiquidityAnalysisProps {
  symbol: string;
}

export function LiquidityAnalysis({ symbol }: LiquidityAnalysisProps) {
  const { data: liquidityData, isLoading } = useQuery<LiquidityAnalysis>({
    queryKey: [`/api/liquidity-analysis/${symbol}`],
    refetchInterval: 10000, // Update every 10 seconds
  });

  const { data: sweepData } = useQuery({
    queryKey: [`/api/liquidity-sweeps/${symbol}`],
    refetchInterval: 5000,
  });

  if (isLoading || !liquidityData) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="text-sm font-medium">Liquidity Analysis</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="animate-pulse">
            <div className="h-4 bg-muted rounded w-3/4 mb-2"></div>
            <div className="h-4 bg-muted rounded w-1/2"></div>
          </div>
        </CardContent>
      </Card>
    );
  }

  const getPoolTypeIcon = (type: string) => {
    switch (type) {
      case 'buy_stops':
      case 'equal_highs':
        return <Target className="h-3 w-3 text-red-500" />;
      case 'sell_stops':
      case 'equal_lows':
        return <Target className="h-3 w-3 text-green-500" />;
      default:
        return <Droplets className="h-3 w-3 text-blue-500" />;
    }
  };

  const getPoolTypeLabel = (type: string) => {
    switch (type) {
      case 'buy_stops': return 'Buy Stops';
      case 'sell_stops': return 'Sell Stops';
      case 'equal_highs': return 'Equal Highs';
      case 'equal_lows': return 'Equal Lows';
      default: return type;
    }
  };

  const getSideColor = (side: string) => {
    switch (side) {
      case 'buy_side': return 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-300';
      case 'sell_side': return 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-300';
      default: return 'bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-300';
    }
  };

  const activePools = liquidityData.pools.slice(0, 5); // Show top 5 pools
  const isGrabInProgress = sweepData?.isGrabInProgress;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-sm font-medium flex items-center gap-2">
          <Droplets className="h-4 w-4" />
          Liquidity Analysis - {symbol}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Liquidity Side Bias */}
        <div className="flex items-center justify-between">
          <span className="text-sm text-muted-foreground">Liquidity Side:</span>
          <Badge className={getSideColor(liquidityData.liquiditySide)}>
            {liquidityData.liquiditySide.replace('_', ' ').toUpperCase()}
          </Badge>
        </div>

        {/* Liquidity Grab Alert */}
        {isGrabInProgress && (
          <div className="flex items-center gap-2 p-2 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg border border-yellow-200 dark:border-yellow-800">
            <AlertTriangle className="h-4 w-4 text-yellow-600" />
            <span className="text-sm text-yellow-800 dark:text-yellow-200">
              Liquidity grab in progress
            </span>
          </div>
        )}

        {/* Next Likely Target */}
        {liquidityData.nextLikelyTarget && (
          <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
            <div className="flex items-center gap-2 mb-2">
              <Target className="h-4 w-4 text-blue-600" />
              <span className="text-sm font-medium text-blue-800 dark:text-blue-200">
                Next Target
              </span>
            </div>
            <div className="space-y-1">
              <div className="flex justify-between text-xs">
                <span className="text-blue-700 dark:text-blue-300">
                  {getPoolTypeLabel(liquidityData.nextLikelyTarget.type)}
                </span>
                <span className="font-mono text-blue-800 dark:text-blue-200">
                  {liquidityData.nextLikelyTarget.price.toFixed(5)}
                </span>
              </div>
              <div className="w-full bg-blue-200 dark:bg-blue-800 rounded-full h-1">
                <div 
                  className="bg-blue-600 h-1 rounded-full" 
                  style={{ width: `${liquidityData.nextLikelyTarget.strength}%` }}
                />
              </div>
            </div>
          </div>
        )}

        {/* Active Liquidity Pools */}
        <div className="space-y-2">
          <h4 className="text-xs font-medium text-muted-foreground">Active Pools</h4>
          {activePools.length > 0 ? (
            <div className="space-y-2">
              {activePools.map((pool) => (
                <div key={pool.id} className="flex items-center justify-between p-2 bg-muted/50 rounded">
                  <div className="flex items-center gap-2">
                    {getPoolTypeIcon(pool.type)}
                    <span className="text-xs">{getPoolTypeLabel(pool.type)}</span>
                    {pool.tested && (
                      <Badge variant="outline" className="text-xs px-1 py-0">
                        Tested
                      </Badge>
                    )}
                  </div>
                  <div className="text-right">
                    <div className="text-xs font-mono">{pool.price.toFixed(5)}</div>
                    <div className="text-xs text-muted-foreground">
                      {pool.strength}% strength
                    </div>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-xs text-muted-foreground text-center py-2">
              No active liquidity pools detected
            </p>
          )}
        </div>

        {/* Recent Sweeps */}
        {liquidityData.recentSweeps.length > 0 && (
          <div className="space-y-2">
            <h4 className="text-xs font-medium text-muted-foreground">Recent Sweeps</h4>
            <div className="space-y-1">
              {liquidityData.recentSweeps.slice(0, 3).map((sweep) => (
                <div key={sweep.id} className="flex items-center justify-between text-xs">
                  <span className="text-muted-foreground">
                    {getPoolTypeLabel(sweep.type)}
                  </span>
                  <span className="font-mono">{sweep.price.toFixed(5)}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}