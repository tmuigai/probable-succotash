import { KillZone } from '@/types/trading';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Clock } from 'lucide-react';

interface KillZonesProps {
  killZones: KillZone[];
}

export function KillZones({ killZones }: KillZonesProps) {
  return (
    <Card className="bg-gray-800 border-gray-700">
      <CardHeader className="pb-3">
        <CardTitle className="text-sm font-semibold text-gray-200 flex items-center">
          <Clock className="w-4 h-4 mr-2 text-yellow-400" />
          Kill Zones
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-2">
        {killZones.map((zone) => (
          <div
            key={zone.name}
            className={`flex justify-between items-center p-2 rounded ${
              zone.active ? 'bg-green-500/20' : 'bg-gray-700'
            }`}
          >
            <span className="text-sm text-white">{zone.name}</span>
            <div className="flex items-center space-x-2">
              <span className="text-xs text-gray-400">
                {String(zone.startHour).padStart(2, '0')}:00-{String(zone.endHour).padStart(2, '0')}:00 {zone.timezone}
              </span>
              <Badge
                variant={zone.active ? "default" : "secondary"}
                className={zone.active ? "bg-green-600 text-white" : ""}
              >
                {zone.active ? 'Active' : 'Inactive'}
              </Badge>
            </div>
          </div>
        ))}
      </CardContent>
    </Card>
  );
}
