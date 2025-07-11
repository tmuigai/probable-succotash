import { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Button } from '@/components/ui/button';
import { Shield, Calculator } from 'lucide-react';
import { useCalculatePosition } from '@/hooks/useMarketData';
import { AccountMetrics } from '@/types/trading';

interface RiskManagementProps {
  accountMetrics: AccountMetrics;
}

export function RiskManagement({ accountMetrics }: RiskManagementProps) {
  const [entryPrice, setEntryPrice] = useState('');
  const [stopLoss, setStopLoss] = useState('');
  const [symbol] = useState('EURUSD'); // Default symbol
  const calculatePosition = useCalculatePosition();

  const handleCalculatePosition = async () => {
    if (!entryPrice || !stopLoss) return;

    try {
      const result = await calculatePosition.mutateAsync({
        entryPrice: parseFloat(entryPrice),
        stopLoss: parseFloat(stopLoss),
        symbol,
      });
      
      // You could show this in a modal or update state to display results
      console.log('Position calculation result:', result);
    } catch (error) {
      console.error('Failed to calculate position:', error);
    }
  };

  const getRiskColor = (percentage: number, max: number) => {
    const ratio = percentage / max;
    if (ratio < 0.5) return 'text-green-400';
    if (ratio < 0.8) return 'text-yellow-400';
    return 'text-red-400';
  };

  return (
    <Card className="bg-gray-800 border-gray-700">
      <CardHeader className="pb-3">
        <CardTitle className="text-sm font-semibold text-gray-200 flex items-center">
          <Shield className="w-4 h-4 mr-2 text-red-400" />
          Risk Management
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="flex justify-between items-center">
          <span className="text-sm text-gray-300">Risk per Trade</span>
          <span className="text-white text-sm font-medium">2%</span>
        </div>
        
        <div className="flex justify-between items-center">
          <span className="text-sm text-gray-300">Daily Risk Limit</span>
          <span className="text-white text-sm font-medium">6%</span>
        </div>
        
        <div className="flex justify-between items-center">
          <span className="text-sm text-gray-300">Used Today</span>
          <span className={`text-sm font-medium ${getRiskColor(accountMetrics?.usedRiskToday || 0, 6)}`}>
            {((accountMetrics?.usedRiskToday ?? 0)).toFixed(1)}%
          </span>
        </div>

        <div className="flex justify-between items-center">
          <span className="text-sm text-gray-300">Current Drawdown</span>
          <span className={`text-sm font-medium ${getRiskColor(accountMetrics?.drawdown || 0, 10)}`}>
            {(accountMetrics?.drawdown || 0).toFixed(1)}%
          </span>
        </div>
        
        <div className="mt-3 p-3 bg-gray-700 rounded">
          <div className="text-xs text-gray-400 mb-2 flex items-center">
            <Calculator className="w-3 h-3 mr-1" />
            Position Size Calculator
          </div>
          
          <div className="space-y-2">
            <div>
              <Label htmlFor="entry-price" className="text-xs text-gray-300">
                Entry Price
              </Label>
              <Input
                id="entry-price"
                type="number"
                step="0.00001"
                placeholder="1.08450"
                value={entryPrice}
                onChange={(e) => setEntryPrice(e.target.value)}
                className="bg-gray-600 border-gray-500 text-white text-xs h-8"
              />
            </div>
            
            <div>
              <Label htmlFor="stop-loss" className="text-xs text-gray-300">
                Stop Loss
              </Label>
              <Input
                id="stop-loss"
                type="number"
                step="0.00001"
                placeholder="1.08300"
                value={stopLoss}
                onChange={(e) => setStopLoss(e.target.value)}
                className="bg-gray-600 border-gray-500 text-white text-xs h-8"
              />
            </div>
            
            <Button
              onClick={handleCalculatePosition}
              disabled={!entryPrice || !stopLoss || calculatePosition.isPending}
              className="w-full bg-blue-600 hover:bg-blue-700 text-white text-xs py-1 h-8"
            >
              {calculatePosition.isPending ? 'Calculating...' : 'Calculate'}
            </Button>
            
            {calculatePosition.data && (
              <div className="text-xs text-white mt-2 p-2 bg-gray-600 rounded">
                <div className="flex justify-between">
                  <span>Lot Size:</span>
                  <span className="font-mono">{calculatePosition.data.lotSize}</span>
                </div>
                <div className="flex justify-between">
                  <span>Risk Amount:</span>
                  <span className="font-mono">${calculatePosition.data.riskAmount.toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                  <span>Potential Profit:</span>
                  <span className="font-mono text-green-400">
                    ${calculatePosition.data.potentialProfit.toFixed(2)}
                  </span>
                </div>
              </div>
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
