import { MarketStructure as MarketStructureType } from '@/types/trading';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { TrendingUp } from 'lucide-react';

interface MarketStructureProps {
  marketStructure: MarketStructureType;
}

export function MarketStructure({ marketStructure }: MarketStructureProps) {
  const getBiasColor = (bias: string) => {
    switch (bias) {
      case 'bullish':
        return 'text-green-400';
      case 'bearish':
        return 'text-red-400';
      default:
        return 'text-yellow-400';
    }
  };

  const getBiasBadge = (bias: string) => {
    switch (bias) {
      case 'bullish':
        return <Badge className="bg-green-500/20 text-green-400 border-green-500/30">Bullish</Badge>;
      case 'bearish':
        return <Badge className="bg-red-500/20 text-red-400 border-red-500/30">Bearish</Badge>;
      default:
        return <Badge className="bg-yellow-500/20 text-yellow-400 border-yellow-500/30">Neutral</Badge>;
    }
  };

  return (
    <Card className="bg-gray-800 border-gray-700">
      <CardHeader className="pb-3">
        <CardTitle className="text-sm font-semibold text-gray-200 flex items-center">
          <TrendingUp className="w-4 h-4 mr-2 text-blue-400" />
          Market Structure
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="flex justify-between items-center">
          <span className="text-sm text-gray-300">HTF Bias</span>
          {getBiasBadge(marketStructure.bias)}
        </div>
        
        <div className="flex justify-between items-center">
          <span className="text-sm text-gray-300">Internal Structure</span>
          <span className={`text-sm font-medium ${getBiasColor(marketStructure.internalStructure)}`}>
            {marketStructure.internalStructure.charAt(0).toUpperCase() + marketStructure.internalStructure.slice(1)}
          </span>
        </div>
        
        <div className="flex justify-between items-center">
          <span className="text-sm text-gray-300">Displacement</span>
          <Badge variant={marketStructure.displacement ? "default" : "secondary"}>
            {marketStructure.displacement ? 'Active' : 'None'}
          </Badge>
        </div>
        
        {marketStructure.lastBOS && (
          <div className="mt-3 p-2 bg-gray-700 rounded">
            <div className="text-xs text-gray-400 mb-1">Last BOS</div>
            <div className="text-sm text-white font-mono">
              {marketStructure.lastBOS.price.toFixed(5)}
            </div>
            <div className="text-xs text-gray-400">
              {new Date(marketStructure.lastBOS.timestamp).toLocaleTimeString()}
            </div>
          </div>
        )}
        
        {marketStructure.lastCHoCH && (
          <div className="mt-2 p-2 bg-gray-700 rounded">
            <div className="text-xs text-gray-400 mb-1">Last CHoCH</div>
            <div className="text-sm text-white font-mono">
              {marketStructure.lastCHoCH.price.toFixed(5)}
            </div>
            <div className="text-xs text-gray-400">
              {new Date(marketStructure.lastCHoCH.timestamp).toLocaleTimeString()}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
