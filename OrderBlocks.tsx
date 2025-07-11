import { OrderBlock } from '@/types/trading';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Box } from 'lucide-react';

interface OrderBlocksProps {
  orderBlocks: OrderBlock[];
  symbol: string;
}

export function OrderBlocks({ orderBlocks, symbol }: OrderBlocksProps) {
  const bullishBlocks = orderBlocks.filter(ob => ob.direction === 'bullish' && ob.valid);
  const bearishBlocks = orderBlocks.filter(ob => ob.direction === 'bearish' && ob.valid);

  return (
    <Card className="bg-gray-800 border-gray-700">
      <CardHeader className="pb-3">
        <CardTitle className="text-sm font-semibold text-gray-200 flex items-center">
          <Box className="w-4 h-4 mr-2 text-purple-400" />
          Order Blocks
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-2">
        {bullishBlocks.length === 0 && bearishBlocks.length === 0 ? (
          <div className="text-center py-4 text-gray-400 text-sm">
            No active order blocks
          </div>
        ) : (
          <>
            {bullishBlocks.map((ob) => (
              <div
                key={ob.id}
                className="border border-green-500/30 rounded p-2 bg-green-500/10"
              >
                <div className="flex justify-between items-center mb-1">
                  <Badge variant="outline" className="text-green-400 border-green-400 text-xs">
                    Bullish OB
                  </Badge>
                  <span className="text-xs text-gray-400">{ob.timeframe}</span>
                </div>
                <div className="text-sm text-white font-mono">
                  {parseFloat(ob.lowPrice).toFixed(5)} - {parseFloat(ob.highPrice).toFixed(5)}
                </div>
                <div className="text-xs text-gray-400">
                  {ob.tested ? 'Tested' : 'Untested'}
                </div>
              </div>
            ))}
            
            {bearishBlocks.map((ob) => (
              <div
                key={ob.id}
                className="border border-red-500/30 rounded p-2 bg-red-500/10"
              >
                <div className="flex justify-between items-center mb-1">
                  <Badge variant="outline" className="text-red-400 border-red-400 text-xs">
                    Bearish OB
                  </Badge>
                  <span className="text-xs text-gray-400">{ob.timeframe}</span>
                </div>
                <div className="text-sm text-white font-mono">
                  {parseFloat(ob.lowPrice).toFixed(5)} - {parseFloat(ob.highPrice).toFixed(5)}
                </div>
                <div className="text-xs text-gray-400">
                  {ob.tested ? 'Tested' : 'Untested'}
                </div>
              </div>
            ))}
          </>
        )}
      </CardContent>
    </Card>
  );
}
