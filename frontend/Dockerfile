FROM node:18-alpine AS base

# Set the project name as a label
LABEL project="PawSense-Frontend"

# Install dependencies only when needed
FROM base AS deps
WORKDIR /app

# Copy package files
COPY package.json package-lock.json* ./

# Install dependencies
RUN npm ci

# Rebuild the source code only when needed
FROM base AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .

# Build the Next.js application
RUN npm run build

# Production image, copy all the files and run the app
FROM base AS runner
WORKDIR /app

ENV NODE_ENV production

# Create a non-root user to run the app
RUN addgroup --system --gid 1001 nodejs && \
    adduser --system --uid 1001 nextjs

# Set proper permissions
COPY --from=builder /app/public ./public
RUN mkdir .next && chown nextjs:nodejs .next

# Copy the production build
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static

# Switch to non-root user
USER nextjs

# Expose the port the app runs on
EXPOSE 3000

# Run the application
CMD ["node", "server.js"] 