import { useEffect, useRef } from 'react';
import { MarketData, OrderBlock, FairValueGap } from '@/types/trading';

interface ChartProps {
  data: MarketData[];
  orderBlocks?: OrderBlock[];
  fvgs?: FairValueGap[];
  title: string;
  height?: number;
  showOverlays?: boolean;
}

export function Chart({ data, orderBlocks = [], fvgs = [], title, height = 400, showOverlays = true }: ChartProps) {
  const chartRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!chartRef.current || !data.length) return;

    // Import Plotly dynamically to avoid SSR issues
    import('plotly.js-dist-min').then((Plotly) => {
      const timestamps = data.map(d => d.timestamp);
      const opens = data.map(d => parseFloat(d.open));
      const highs = data.map(d => parseFloat(d.high));
      const lows = data.map(d => parseFloat(d.low));
      const closes = data.map(d => parseFloat(d.close));

      const traces: any[] = [
        {
          x: timestamps,
          open: opens,
          high: highs,
          low: lows,
          close: closes,
          type: 'candlestick',
          name: title,
          increasing: { line: { color: '#10B981' } },
          decreasing: { line: { color: '#EF4444' } },
          showlegend: false,
        }
      ];

      // Add Order Blocks as shapes
      const shapes: any[] = [];
      
      if (showOverlays && orderBlocks.length > 0) {
        orderBlocks.forEach((ob) => {
          shapes.push({
            type: 'rect',
            x0: ob.timestamp,
            x1: timestamps[timestamps.length - 1], // Extend to current time
            y0: parseFloat(ob.lowPrice),
            y1: parseFloat(ob.highPrice),
            fillcolor: ob.direction === 'bullish' ? 'rgba(16, 185, 129, 0.2)' : 'rgba(239, 68, 68, 0.2)',
            line: {
              color: ob.direction === 'bullish' ? '#10B981' : '#EF4444',
              width: 1,
              dash: 'dot',
            },
            layer: 'below',
          });
        });
      }

      // Add Fair Value Gaps as shapes
      if (showOverlays && fvgs.length > 0) {
        fvgs.forEach((fvg) => {
          if (!fvg.filled) {
            shapes.push({
              type: 'rect',
              x0: fvg.timestamp,
              x1: timestamps[timestamps.length - 1],
              y0: fvg.lowPrice,
              y1: fvg.highPrice,
              fillcolor: fvg.direction === 'bullish' ? 'rgba(59, 130, 246, 0.15)' : 'rgba(245, 158, 11, 0.15)',
              line: {
                color: fvg.direction === 'bullish' ? '#3B82F6' : '#F59E0B',
                width: 1,
                dash: 'dashdot',
              },
              layer: 'below',
            });
          }
        });
      }

      const layout = {
        title: {
          text: title,
          font: { color: '#F8FAFC', size: 14 },
        },
        xaxis: {
          type: 'date',
          color: '#94A3B8',
          gridcolor: '#374151',
          showgrid: true,
        },
        yaxis: {
          color: '#94A3B8',
          gridcolor: '#374151',
          showgrid: true,
          side: 'right',
        },
        plot_bgcolor: '#1F2937',
        paper_bgcolor: '#1F2937',
        font: { color: '#F8FAFC', size: 11 },
        margin: { l: 20, r: 60, t: 40, b: 40 },
        shapes,
        dragmode: 'pan',
        scrollZoom: true,
      };

      const config = {
        responsive: true,
        displayModeBar: false,
        doubleClick: 'reset',
      };

      Plotly.newPlot(chartRef.current, traces, layout, config);
    });

    return () => {
      if (chartRef.current) {
        // Clean up the chart
        import('plotly.js-dist-min').then((Plotly) => {
          Plotly.purge(chartRef.current!);
        });
      }
    };
  }, [data, orderBlocks, fvgs, title, showOverlays]);

  return (
    <div 
      ref={chartRef} 
      className="w-full bg-gray-900 rounded"
      style={{ height: `${height}px` }}
    />
  );
}
