# Cloud Deployment Guide

Deploy the Biometric Tracking System on AWS, GCP, or Azure.

## Prerequisites

- Docker and Docker Compose installed on the cloud VM
- Open ports: 80 (HTTP), 443 (HTTPS), 8000 (API), 8501 (Dashboard)
- At least 8GB RAM, 4 vCPUs recommended

## Quick Deploy

### 1. SSH into your cloud VM

```bash
ssh user@your-vm-ip
```

### 2. Clone the repository

```bash
git clone https://github.com/your-username/biometric-tracking-system.git
cd biometric-tracking-system
```

### 3. Create environment file

```bash
cp .env.example .env
nano .env  # Edit with your configuration
```

### 4. Deploy with Docker Compose

```bash
# For cloud deployment with nginx
docker-compose -f docker-compose.yml -f deploy/cloud/docker-compose.cloud.yml up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f
```

## Cloud Provider Specific Guides

### AWS EC2

1. Launch an EC2 instance (Ubuntu 22.04 LTS recommended)
2. Configure Security Group to allow inbound:
   - Port 80 (HTTP)
   - Port 443 (HTTPS)
   - Port 22 (SSH)
3. SSH and install Docker:
   ```bash
   sudo apt update
   sudo apt install docker.io docker-compose -y
   sudo usermod -aG docker $USER
   newgrp docker
   ```
4. Continue with Quick Deploy steps

### Google Cloud Platform (GCP)

1. Create a Compute Engine VM instance
2. Allow HTTP/HTTPS traffic in firewall rules
3. SSH via Google Cloud Console or:
   ```bash
   gcloud compute ssh --zone "your-zone" "your-instance"
   ```
4. Install Docker and deploy

### Azure Virtual Machine

1. Create a Linux VM (Ubuntu recommended)
2. Add inbound port rules for 80, 443
3. SSH via Azure portal or:
   ```bash
   ssh azureuser@your-vm-ip
   ```
4. Install Docker and deploy

## Managing the Deployment

```bash
# Check service health
docker-compose ps

# View logs
docker-compose logs -f backend
docker-compose logs -f dashboard

# Restart a service
docker-compose restart backend

# Update and redeploy
git pull
docker-compose -f docker-compose.yml -f deploy/cloud/docker-compose.cloud.yml up -d --build

# Stop all services
docker-compose down
```

## SSL/HTTPS Setup (Optional)

1. Place your SSL certificates:
   ```bash
   mkdir -p deploy/cloud/ssl
   cp your-cert.pem deploy/cloud/ssl/cert.pem
   cp your-key.pem deploy/cloud/ssl/key.pem
   ```

2. Uncomment the HTTPS server block in `nginx.conf`

3. Restart nginx:
   ```bash
   docker-compose restart nginx
   ```

## Monitoring

```bash
# Resource usage
docker stats

# Health checks
curl http://localhost:8000/health
curl http://localhost:8501/healthz
```

## Troubleshooting

### Services not starting
```bash
docker-compose logs [service-name]
```

### Port already in use
```bash
sudo netstat -tulpn | grep :80
sudo kill <pid>
```

### Out of memory
Increase VM resources or reduce `deploy.resources` limits in `docker-compose.cloud.yml`
