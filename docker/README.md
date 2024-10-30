# Streamlit Meetup Recommendation System - Docker Deployment

This directory contains files necessary for deploying the Streamlit Meetup Recommendation System with Docker. This guide provides instructions to build and run the application using Docker and Docker Compose.

## Prerequisites

Make sure the following tools are installed on your system:
- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/)

## Directory Structure
The directory structure for this project should look like this 


```plaintext
project-root/
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── your_script.py
├── requirements.txt
└── .env
```

## Environment Setup

1. **Environment Variables**
    - Ensure there is an `.env` file in the project root directory. 
    - The file should contain the necessary environment variables, such as the OpenAI API key
   - `OPENAI_API_KEY=your_openai_api_key_here`
   

2. **Python Dependencies**: 
   - The `requirements.txt` file should include all Python packages needed for the project. This will be used during the Docker build process.



## Instructions

### 1. Build and Run the Application

To build and run the application, use the following commands

1. **Navigate to the Docker directory**
```bash
cd docker
```

2. **Build and run the Docker container**
```bash
docker-compose up --build
```

3. **Access the application: Once the container is running, open your browser and go to:**
```bash
http://localhost:8501
```

4. **Stopping the Application**
To stop the application and remove the container, press Ctrl + C in the terminal where it’s running, or run:

```bash
docker-compose down
```

