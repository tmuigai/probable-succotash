import { useState, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { TrendingUp, TrendingDown, CircleDot } from 'lucide-react';

interface PO3Result {
  phase: string;
  accumulation: boolean;
  manipulation: boolean;
  distribution: 'bullish' | 'bearish' | 'none';
  signal: 'buy' | 'sell' | null;
  confidence: number;
  combinedSignal: 'buy' | 'sell' | null;
}

interface PO3AnalysisProps {
  symbol: string;
}

export function PO3Analysis({ symbol }: PO3AnalysisProps) {
  const { data: po3Data, isLoading } = useQuery<PO3Result>({
    queryKey: [`/api/po3-analysis/${symbol}`],
    refetchInterval: 5000, // Update every 5 seconds
  });

  if (isLoading || !po3Data) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="text-sm font-medium">PO3 Analysis</CardTitle>
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

  const getPhaseColor = (phase: string) => {
    switch (phase) {
      case 'accumulation': return 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-300';
      case 'manipulation': return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-300';
      case 'bullish': return 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-300';
      case 'bearish': return 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-300';
      default: return 'bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-300';
    }
  };

  const getSignalIcon = (signal: string | null) => {
    if (signal === 'buy') return <TrendingUp className="h-4 w-4 text-green-500" />;
    if (signal === 'sell') return <TrendingDown className="h-4 w-4 text-red-500" />;
    return <CircleDot className="h-4 w-4 text-gray-400" />;
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-sm font-medium">PO3 Analysis - {symbol}</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Current Phase */}
        <div className="flex items-center justify-between">
          <span className="text-sm text-muted-foreground">Phase:</span>
          <Badge className={getPhaseColor(po3Data.phase)}>
            {po3Data.phase.charAt(0).toUpperCase() + po3Data.phase.slice(1)}
          </Badge>
        </div>

        {/* Phase Indicators */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-xs text-muted-foreground">Accumulation</span>
            <div className={`w-3 h-3 rounded-full ${po3Data.accumulation ? 'bg-blue-500' : 'bg-gray-300'}`} />
          </div>
          <div className="flex items-center justify-between">
            <span className="text-xs text-muted-foreground">Manipulation</span>
            <div className={`w-3 h-3 rounded-full ${po3Data.manipulation ? 'bg-yellow-500' : 'bg-gray-300'}`} />
          </div>
          <div className="flex items-center justify-between">
            <span className="text-xs text-muted-foreground">Distribution</span>
            <div className={`w-3 h-3 rounded-full ${
              po3Data.distribution === 'bullish' ? 'bg-green-500' : 
              po3Data.distribution === 'bearish' ? 'bg-red-500' : 'bg-gray-300'
            }`} />
          </div>
        </div>

        {/* Signal Information */}
        {po3Data.signal && (
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-sm text-muted-foreground">Signal:</span>
              <div className="flex items-center gap-2">
                {getSignalIcon(po3Data.signal)}
                <span className="text-sm font-medium capitalize">{po3Data.signal}</span>
              </div>
            </div>
            
            <div className="space-y-1">
              <div className="flex items-center justify-between text-xs">
                <span>Confidence</span>
                <span>{po3Data.confidence.toFixed(1)}%</span>
              </div>
              <Progress value={po3Data.confidence} className="h-2" />
            </div>
          </div>
        )}

        {/* Combined Signal (Multi-Symbol Analysis) */}
        {po3Data.combinedSignal && (
          <div className="border-t pt-2">
            <div className="flex items-center justify-between">
              <span className="text-sm text-muted-foreground">Combined Signal:</span>
              <div className="flex items-center gap-2">
                {getSignalIcon(po3Data.combinedSignal)}
                <span className="text-sm font-medium capitalize">{po3Data.combinedSignal}</span>
                <Badge variant="outline" className="text-xs">Confirmed</Badge>
              </div>
            </div>
          </div>
        )}

        {/* No Signal State */}
        {!po3Data.signal && (
          <div className="text-center py-2">
            <CircleDot className="h-8 w-8 text-gray-400 mx-auto mb-2" />
            <p className="text-xs text-muted-foreground">No active PO3 signal</p>
          </div>
        )}
      </CardContent>
    </Card>
  );
}