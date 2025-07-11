import { Signal } from '@/types/trading';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { AlertTriangle, Play } from 'lucide-react';
import { useExecuteSignal } from '@/hooks/useMarketData';
import { useToast } from '@/hooks/use-toast';

interface SignalsProps {
  signals: Signal[];
}

export function Signals({ signals }: SignalsProps) {
  const executeSignal = useExecuteSignal();
  const { toast } = useToast();

  const activeSignals = signals.filter(signal => signal.status === 'active');

  const handleExecuteSignal = async (signalId: number) => {
    try {
      await executeSignal.mutateAsync(signalId);
      toast({
        title: "Signal Executed",
        description: "Trade has been opened successfully",
      });
    } catch (error) {
      toast({
        title: "Execution Failed",
        description: error instanceof Error ? error.message : "Failed to execute signal",
        variant: "destructive",
      });
    }
  };

  const getDirectionColor = (direction: string) => {
    return direction === 'buy' ? 'text-green-400' : 'text-red-400';
  };

  const getDirectionBadge = (direction: string) => {
    return direction === 'buy' 
      ? <Badge className="bg-green-500/20 text-green-400 border-green-500/30">BUY SETUP</Badge>
      : <Badge className="bg-red-500/20 text-red-400 border-red-500/30">SELL SETUP</Badge>;
  };

  return (
    <Card className="bg-gray-800 border-gray-700">
      <CardHeader className="pb-3">
        <CardTitle className="text-sm font-semibold text-gray-200 flex items-center">
          <AlertTriangle className="w-4 h-4 mr-2 text-green-400" />
          ICT Signals
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        {activeSignals.length === 0 ? (
          <div className="text-center py-4 text-gray-400 text-sm">
            No active signals
          </div>
        ) : (
          activeSignals.map((signal) => (
            <div
              key={signal.id}
              className={`border rounded p-3 ${
                signal.direction === 'buy' 
                  ? 'border-green-500/30 bg-green-500/10' 
                  : 'border-red-500/30 bg-red-500/10'
              }`}
            >
              <div className="flex justify-between items-center mb-2">
                {getDirectionBadge(signal.direction)}
                <span className="text-xs text-gray-400">
                  {new Date(signal.timestamp).toLocaleTimeString()}
                </span>
              </div>
              
              <div className="text-xs text-gray-300 space-y-1 mb-3">
                <div className="flex justify-between">
                  <span>Entry:</span>
                  <span className="font-mono">{parseFloat(signal.entryPrice).toFixed(5)}</span>
                </div>
                <div className="flex justify-between">
                  <span>SL:</span>
                  <span className="font-mono">{parseFloat(signal.stopLoss).toFixed(5)}</span>
                </div>
                <div className="flex justify-between">
                  <span>TP:</span>
                  <span className="font-mono">{parseFloat(signal.takeProfit).toFixed(5)}</span>
                </div>
                <div className="flex justify-between">
                  <span>R:R:</span>
                  <span className={getDirectionColor(signal.direction)}>
                    {parseFloat(signal.riskReward).toFixed(1)}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>Confidence:</span>
                  <span className="text-yellow-400">
                    {(parseFloat(signal.confidence) * 100).toFixed(0)}%
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>Type:</span>
                  <span className="text-blue-400">{signal.signalType}</span>
                </div>
              </div>
              
              <Button
                onClick={() => handleExecuteSignal(signal.id)}
                disabled={executeSignal.isPending}
                className={`w-full text-xs py-1 ${
                  signal.direction === 'buy'
                    ? 'bg-green-600 hover:bg-green-700'
                    : 'bg-red-600 hover:bg-red-700'
                } text-white`}
              >
                <Play className="w-3 h-3 mr-1" />
                {executeSignal.isPending ? 'Executing...' : 'Execute Trade'}
              </Button>
            </div>
          ))
        )}
      </CardContent>
    </Card>
  );
}
