## Question 1

* Install `uv`:
```sh
python -V
```
`Python 3.12.1`

    pip install uv

* What's the version of uv you installed? Use `--version` to find out:
```sh
uv --version
```
`uv 0.9.5`

## Initialize an empty uv project

    uv init

## Question 2

* Use uv to install Scikit-Learn version 1.6.1:

    uv add scikit-learn==1.6.1

* What's the first hash for Scikit-Learn you get in the lock file? Include the entire string starting with sha256, don't include quotes:
`sha256:3faa5c39054b2f03ca547da9b2f52fde67c06240c31853f306aea97f13647b55`

## Models

> **Note**: You don't need to train the model. This code is just for your reference.

    wget https://github.com/DataTalksClub/machine-learning-zoomcamp/raw/refs/heads/master/cohorts/2025/05-deployment/pipeline_v1.bin

## Question 3

* Write a script for loading the pipeline with pickle. Score this record:
```json
{
    "lead_source": "paid_ads",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0
}
```

What's the probability that this lead will convert? 

    python predict.py

`0.534`

* 0.333
* **0.533**
* 0.733
* 0.933

## Question 4

Now let's serve this model as a web service

* Install FastAPI
* Write FastAPI code for serving the model
* Now score this client using `requests`:

```python
url = "YOUR_URL"
client = {
    "lead_source": "organic_search",
    "number_of_courses_viewed": 4,
    "annual_income": 80304.0
}
requests.post(url, json=client).json()
```

    pip install fastapi requests uvicorn
    python predict.py

What's the probability that this client will get a subscription?

    python test.py
    
```Conversion probability: 0.534```

```Lead is likely to convert, follow up```

* 0.334
* **0.534**
* 0.734
* 0.934

## Question 5

Download the base image `agrigorev/zoomcamp-model:2025`. You can easily make it by using [docker pull](https://docs.docker.com/engine/reference/commandline/pull/) command.

So what's the size of this base image?

    docker pull agrigorev/zoomcamp-model:2025

    docker images

* 45 MB
* **121 MB**
* 245 MB
* 330 MB

You can get this information when running `docker images` - it'll be in the "SIZE" column.

## Dockerfile

Now create your own `Dockerfile` based on the image we prepared.

It should start like that:

```docker
FROM agrigorev/zoomcamp-model:2025
# add your stuff here
```

Now complete it:

* Install all the dependencies from pyproject.toml
* Copy your FastAPI script
* Run it with uvicorn 

After that, you can build your docker image.

## Question 6

Let's run your docker container!

After running it, score this client once again:

```python
url = "http://localhost:9696/predict"
client = {
    "lead_source": "organic_search",
    "number_of_courses_viewed": 4,
    "annual_income": 80304.0
}
requests.post(url, json=client).json()
```

### Steps:
1. Build the Docker image:
    ```bash
    docker build -t lead-conversion-service .
    ```

2. Check if port 9696 is in use:
    ```bash
    sudo lsof -i :9696
    ```

3. If necessary, stop the process using the port:
    ```bash
    sudo kill -9 <PID>
    ```

4. Run the Docker container:
    ```bash
    docker run -it --rm -p 9696:9696 lead-conversion-service
    ```

5. Test the `/predict` endpoint:
    ```bash
    curl -X POST -H "Content-Type: application/json" \
    -d '{"lead_source": "organic_search", "number_of_courses_viewed": 4, "annual_income": 80304.0}' \
    http://localhost:9696/predict
    ```

### Output:
```json
{"conversion_probability": 0.5340417283801275, "convert": true}
```

* 0.39
* **0.59**
* 0.79
* 0.99