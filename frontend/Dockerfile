FROM node:16-alpine as build

WORKDIR /app

# Install curl for healthcheck
RUN apk --no-cache add curl

# Add build arguments
ARG REACT_APP_API_URL
ARG REACT_APP_API_ENDPOINT
ENV REACT_APP_API_URL=${REACT_APP_API_URL}
ENV REACT_APP_API_ENDPOINT=${REACT_APP_API_ENDPOINT}

# Copy package.json and install dependencies
COPY package*.json ./
RUN npm install --production

# Copy the rest of the code and build
COPY . .
RUN npm run build

# Production environment
FROM nginx:alpine
COPY --from=build /app/build /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Add curl for healthcheck
RUN apk --no-cache add curl

# Create a simple health endpoint
RUN echo "healthy" > /usr/share/nginx/html/health

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
