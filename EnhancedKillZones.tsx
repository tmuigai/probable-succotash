import { useQuery } from '@tanstack/react-query';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Clock, AlertCircle, CheckCircle, Timer } from 'lucide-react';

interface KillZone {
  name: string;
  startHour: number;
  endHour: number;
  timezone: string;
  active: boolean;
  priority: 'high' | 'medium' | 'low';
  description: string;
}

interface KillZoneAnalysis {
  currentZone: KillZone | null;
  nextZone: KillZone | null;
  timeToNext: number;
  allZones: KillZone[];
  recommendation: 'trade' | 'wait' | 'avoid';
}

interface SessionOverlap {
  session: string;
  active: boolean;
  priority: string;
}

export function EnhancedKillZones() {
  const { data: killZoneData, isLoading } = useQuery<KillZoneAnalysis>({
    queryKey: ['/api/kill-zones-enhanced'],
    refetchInterval: 60000, // Update every minute
  });

  const { data: optimalTimeData } = useQuery({
    queryKey: ['/api/optimal-time'],
    refetchInterval: 60000,
  });

  const { data: sessionOverlaps } = useQuery<SessionOverlap[]>({
    queryKey: ['/api/session-overlaps'],
    refetchInterval: 60000,
  });

  if (isLoading || !killZoneData) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="text-sm font-medium">Kill Zones</CardTitle>
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

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'high': return 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-300';
      case 'medium': return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-300';
      case 'low': return 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-300';
      default: return 'bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-300';
    }
  };

  const getRecommendationIcon = (recommendation: string) => {
    switch (recommendation) {
      case 'trade': return <CheckCircle className="h-4 w-4 text-green-500" />;
      case 'wait': return <Timer className="h-4 w-4 text-yellow-500" />;
      case 'avoid': return <AlertCircle className="h-4 w-4 text-red-500" />;
      default: return <Clock className="h-4 w-4 text-gray-400" />;
    }
  };

  const getRecommendationColor = (recommendation: string) => {
    switch (recommendation) {
      case 'trade': return 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-300';
      case 'wait': return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-300';
      case 'avoid': return 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-300';
      default: return 'bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-300';
    }
  };

  const formatTime = (minutes: number): string => {
    const hours = Math.floor(minutes / 60);
    const mins = minutes % 60;
    if (hours > 0) {
      return `${hours}h ${mins}m`;
    }
    return `${mins}m`;
  };

  const formatHour = (hour: number): string => {
    return `${hour.toString().padStart(2, '0')}:00 UTC`;
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-sm font-medium flex items-center gap-2">
          <Clock className="h-4 w-4" />
          Kill Zones & Sessions
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Current Status */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-sm text-muted-foreground">Status:</span>
            <div className="flex items-center gap-2">
              {getRecommendationIcon(killZoneData.recommendation)}
              <Badge className={getRecommendationColor(killZoneData.recommendation)}>
                {killZoneData.recommendation.toUpperCase()}
              </Badge>
            </div>
          </div>

          {optimalTimeData?.isOptimalTime && (
            <div className="p-2 bg-green-50 dark:bg-green-900/20 rounded-lg border border-green-200 dark:border-green-800">
              <div className="flex items-center gap-2">
                <CheckCircle className="h-4 w-4 text-green-600" />
                <span className="text-sm text-green-800 dark:text-green-200 font-medium">
                  Optimal Trading Time
                </span>
              </div>
            </div>
          )}
        </div>

        {/* Current Zone */}
        {killZoneData.currentZone ? (
          <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-medium text-blue-800 dark:text-blue-200">
                {killZoneData.currentZone.name}
              </span>
              <Badge className={getPriorityColor(killZoneData.currentZone.priority)}>
                {killZoneData.currentZone.priority.toUpperCase()}
              </Badge>
            </div>
            <p className="text-xs text-blue-700 dark:text-blue-300">
              {killZoneData.currentZone.description}
            </p>
            <div className="text-xs text-blue-600 dark:text-blue-400 mt-1">
              {formatHour(killZoneData.currentZone.startHour)} - {formatHour(killZoneData.currentZone.endHour)}
            </div>
          </div>
        ) : (
          <div className="p-3 bg-gray-50 dark:bg-gray-900/20 rounded-lg border border-gray-200 dark:border-gray-800">
            <span className="text-sm text-gray-600 dark:text-gray-400">
              No active kill zone
            </span>
          </div>
        )}

        {/* Next Zone */}
        {killZoneData.nextZone && (
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-sm text-muted-foreground">Next Zone:</span>
              <span className="text-sm font-medium">{killZoneData.nextZone.name}</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-xs text-muted-foreground">Time to start:</span>
              <span className="text-xs font-mono">{formatTime(killZoneData.timeToNext)}</span>
            </div>
          </div>
        )}

        {/* Session Overlaps */}
        {sessionOverlaps && sessionOverlaps.length > 0 && (
          <div className="space-y-2">
            <h4 className="text-xs font-medium text-muted-foreground">Session Overlaps</h4>
            <div className="space-y-1">
              {sessionOverlaps.map((overlap, index) => (
                <div key={index} className="flex items-center justify-between p-2 bg-orange-50 dark:bg-orange-900/20 rounded border border-orange-200 dark:border-orange-800">
                  <span className="text-xs text-orange-800 dark:text-orange-200">
                    {overlap.session}
                  </span>
                  <Badge className="text-xs px-1 py-0 bg-orange-100 text-orange-800 dark:bg-orange-900 dark:text-orange-300">
                    {overlap.priority.replace('-', ' ').toUpperCase()}
                  </Badge>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* All Zones Status */}
        <div className="space-y-2">
          <h4 className="text-xs font-medium text-muted-foreground">All Zones</h4>
          <div className="space-y-1">
            {killZoneData.allZones.map((zone, index) => (
              <div key={index} className="flex items-center justify-between text-xs">
                <div className="flex items-center gap-2">
                  <div className={`w-2 h-2 rounded-full ${zone.active ? 'bg-green-500' : 'bg-gray-300'}`} />
                  <span className={zone.active ? 'font-medium' : 'text-muted-foreground'}>
                    {zone.name}
                  </span>
                </div>
                <span className="text-muted-foreground font-mono">
                  {formatHour(zone.startHour)}-{formatHour(zone.endHour)}
                </span>
              </div>
            ))}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}