# ICT Trading Dashboard

## Overview

This is a full-stack web application built for trading analysis using Inner Circle Trader (ICT) methodology. The application provides real-time market data analysis, signal generation, and trade management capabilities with a focus on Forex markets.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

The application follows a modern full-stack architecture with:

- **Frontend**: React with TypeScript, using Vite as the build tool
- **Backend**: Express.js with TypeScript
- **Database**: PostgreSQL with Drizzle ORM
- **Real-time Communication**: WebSocket for live data streaming
- **UI Framework**: Tailwind CSS with Radix UI components (shadcn/ui)
- **State Management**: TanStack Query for server state management

## Key Components

### Frontend Architecture

- **React SPA**: Single-page application with client-side routing using Wouter
- **Component Library**: Comprehensive set of reusable UI components from shadcn/ui
- **State Management**: React Query for API state, local React state for UI
- **Styling**: Tailwind CSS with CSS variables for theming
- **TypeScript**: Full type safety across the application

### Backend Architecture

- **Express Server**: RESTful API with WebSocket support
- **Service Layer**: Modular services for ICT analysis, market data, and risk management
- **Data Storage**: In-memory storage with interface for future database integration
- **Real-time Updates**: WebSocket server for live price feeds and notifications

### Database Schema

The application uses Drizzle ORM with PostgreSQL, defining several key entities:

- **Users**: Basic user management
- **Trades**: Complete trade lifecycle tracking
- **Market Data**: OHLCV candlestick data storage
- **Order Blocks**: ICT-specific order block identification
- **Signals**: Trading signal generation and tracking

### ICT Analysis Services

- **Market Structure Analysis**: Identifies break of structure (BOS) and change of character (CHoCH)
- **Order Block Detection**: Finds institutional order blocks using ICT methodology
- **Fair Value Gap (FVG) Identification**: Detects and tracks imbalance areas
- **Kill Zone Monitoring**: Tracks optimal trading sessions (London/New York)
- **Risk Management**: Position sizing and risk calculations
- **PO3 Analysis**: Power of Three detection for accumulation, manipulation, and distribution phases
- **Advanced Liquidity Analysis**: Liquidity pool detection, sweep analysis, and institutional order flow
- **Enhanced Kill Zones**: Comprehensive session analysis with overlaps and optimal timing
- **Multi-Symbol Correlation**: Cross-pair analysis for signal confirmation
- **Precision Entry Detection**: Advanced signal validation with confidence scoring

## Data Flow

1. **Market Data Ingestion**: Real-time price data simulation (ready for live feeds)
2. **Analysis Pipeline**: ICT analysis services process market data
3. **Signal Generation**: Automated signal creation based on ICT patterns
4. **WebSocket Broadcasting**: Real-time updates sent to connected clients
5. **Trade Execution**: Manual/automated trade placement with risk management
6. **Performance Tracking**: Trade analysis and portfolio metrics

## External Dependencies

### Core Framework Dependencies
- React ecosystem (React, React DOM, React Query)
- Express.js for backend
- Drizzle ORM with PostgreSQL support
- WebSocket (ws) for real-time communication

### UI and Styling
- Tailwind CSS for styling
- Radix UI primitives for accessible components
- Lucide React for icons
- Custom theme system with CSS variables

### Development Tools
- Vite for build tooling and development server
- TypeScript for type safety
- ESBuild for production builds
- Replit-specific plugins for development environment

### Trading-Specific Libraries
- Date manipulation (date-fns)
- Chart visualization capabilities (Plotly.js integration ready)
- Form handling (React Hook Form with Zod validation)

## Deployment Strategy

The application is configured for multiple deployment scenarios:

### Development
- Vite dev server with hot module replacement
- Express server with auto-reload
- In-memory storage for rapid prototyping
- WebSocket integration with development tools

### Production
- Vite build generates optimized static assets
- ESBuild bundles server code
- Static file serving through Express
- Environment-based configuration

### Database Migration
- Drizzle migrations stored in `/migrations`
- PostgreSQL connection via environment variables
- Neon Database serverless integration ready
- Schema definitions shared between client and server

## Latest Updates (July 8, 2025)

Added comprehensive advanced ICT features based on professional trading bot implementation:

### New Services Added:
- **PO3 Analysis Service**: Real-time Power of Three phase detection with multi-symbol correlation
- **Advanced Kill Zone Service**: Enhanced session analysis with optimal timing recommendations
- **Liquidity Analysis Service**: Institutional liquidity pool detection and sweep monitoring

### New API Endpoints:
- `/api/po3-analysis/:symbol` - Power of Three analysis for any symbol
- `/api/kill-zones-enhanced` - Advanced kill zone analysis with timing
- `/api/liquidity-analysis/:symbol` - Liquidity pool and sweep detection
- `/api/session-overlaps` - Trading session overlap analysis
- `/api/optimal-time` - Optimal trading time recommendations

### New Frontend Components:
- **PO3Analysis Component**: Displays accumulation, manipulation, distribution phases
- **LiquidityAnalysis Component**: Shows liquidity pools, sweeps, and targets
- **EnhancedKillZones Component**: Advanced session timing and recommendations

The architecture prioritizes modularity, allowing for easy extension of ICT analysis techniques and integration with live trading platforms. The separation of concerns between analysis services, data management, and UI components enables rapid development and testing of new trading strategies. The platform now includes all major ICT Smart Money concepts with real-time analysis and professional-grade features.