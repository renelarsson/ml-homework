## Homework

In this homework, we'll deploy the lead scoring model from the homework 5.

We already have a docker image for this model - we'll use it for 
deploying the model to Kubernetes.


## Building the image

Clone the course repo if you haven't:

```
git clone https://github.com/DataTalksClub/machine-learning-zoomcamp.git
```

Go to the `course-zoomcamp/cohorts/2025/05-deployment/homework` folder and 
execute the following:


```bash
docker build -f Dockerfile_full -t zoomcamp-model:3.13.10-hw10 .
```


## Question 1

Run it to test that it's working locally:

```bash
docker run -it --rm -p 9696:9696 zoomcamp-model:3.13.10-hw10
```

And in another terminal, execute `q6_test.py` file:

```bash
python q6_test.py
```

You should see this:

```python
{'conversion_probability': <value>, 'conversion': False}
```

Here `<value>` is the probability of getting a subscription. You need to choose the right one.

`{'conversion_probability': 0.49999999999842815, 'conversion': False}`

* 0.29
* **`0.49`**
* 0.69
* 0.89

Now you can stop the container running in Docker.

```bash
docker ps
docker stop <container_id>
```

## Installing `kubectl` and `kind`

You need to install:

* `kubectl` - https://kubernetes.io/docs/tasks/tools/ (you might already have it - check before installing)
* `kind` - https://kind.sigs.k8s.io/docs/user/quick-start/

- `kubectl` is the command-line tool used to interact with Kubernetes clusters. 
- `kind` (short for **Kubernetes IN Docker**) is a tool for running local Kubernetes clusters using Docker containers. It is primarily designed for testing and development purposes, allowing you to create lightweight, multi-node Kubernetes clusters on your local machine without needing a full cloud-based Kubernetes setup.

## Question 2

What's the version of `kind` that you have? 

Use `kind --version` to find out.

Check if `${HOME}/bin` is already in your PATH, if not, add it:

```bash
echo $PATH | grep -q "${HOME}/bin" && echo "Exists in PATH" || echo "Not in PATH"
export PATH="${PATH}:${HOME}/bin"
```

Make it permanent (Optional):

```bash
echo 'export PATH="${PATH}:${HOME}/bin"' >> ~/.bashrc
source ~/.bashrc
```

Download the kind binary and make it executable:

```bash
mkdir -p ${HOME}/bin
curl -Lo ${HOME}/bin/kind https://kind.sigs.k8s.io/dl/v0.20.0/kind-linux-amd64
chmod +x ${HOME}/bin/kind
```

- The purpose of these steps is to install the kind tool locally in the home directory without requiring root or system-wide permissions. By placing the binary in ${HOME}/bin, you can add this directory to your PATH environment variable, making the kind command available globally in your terminal.

Verify installation:
```bash
kind version
```
`kind v0.20.0 go1.20.4 linux/amd64`

## Creating a cluster

- A **Kubernetes cluster** is a set of machines (nodes) that work together to run and manage containerized applications. It provides a platform to automate the deployment, scaling, and operation of application containers across multiple hosts.

Now let's create a cluster with `kind`:

```bash
kind create cluster
```

And check with `kubectl` that it was successfully created:

```bash
kubectl cluster-info
```


## Question 3

- A **cluster** is a Kubernetes deployment.
- A **Pod** is the smallest deployable unit in Kubernetes. It represents a single instance of a running process in the cluster. Pods are used to run application workloads and can contain one or more containers that share the same IP address (Networking) or volumes (Storage).
- A **Node** is a physical or virtual machine in the Kubernetes cluster that runs workloads (a server such as a local computer or an EC2 instance). It ensures containers in pods are running, it manages networking for pods, and runs containers. Every cluster contains at least one node.
- A **Deployment** is a higher-level abstraction (a group of pods all running the same image and config) that manages pods and ensures the desired state of the application. The pods may be distributed in different nodes. They are used to manage stateless applications, and handle replicas, i.e. ensure a specified number of pods are running, updated and replaced automatically. 
- A **Service** is an abstraction (an entrypoint that serves as a middleman between the user and the deployment) that provides a stable network endpoint (IP and DNS) to access a set of pods and expose your application to other pods or external clients. The Service receives requests from users and routes them to available pods within the deployment, which sends a reply to the user (a pod from a different deployment or an external application).   Services can therefore be **external or internal**:
    - **LoadBalancer**: Exposes the service externally using a cloud provider's load balancer.
    - **ClusterIP**: IDefault; exposes the service within the cluster.
- *An **Ingress** is a resource that exposes HTTP/HTTPS routes from outside the cluster to services within the cluster.*

What's the smallest deployable computing unit that we can create and manage 
in Kubernetes (`kind` in our case)?

* Node
* **`Pod`**
* Deployment
* Service


## Question 4

Now let's test if everything works. Use `kubectl` to get the list of running services.

Check the service:

```bash
kubectl get services
```

What's the `Type` of the service that is already running there?

* NodePort
* **`ClusterIP`**
* ExternalName
* LoadBalancer


## Question 5

To be able to use the docker image we previously created (`zoomcamp-model:3.13.10-hw10`),
we need to register it with `kind`.

What's the command we need to run for that?

```bash
kind load docker-image zoomcamp-model:3.13.10-hw10
```

* kind create cluster
* kind build node-image
* **`kind load docker-image`**
* kubectl apply


## Question 6

Now let's create a deployment config (e.g. `deployment.yaml`):

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: subscription
spec:
  selector:
    matchLabels:
      app: subscription
  replicas: 1
  template:
    metadata:
      labels:
        app: subscription
    spec:
      containers:
      - name: subscription
        image: <Image> # The application's Docker image
        resources:
          requests:
            memory: "64Mi"
            cpu: "100m"            
          limits:
            memory: <Memory>
            cpu: <CPU>
        ports:
        - containerPort: <Port> # the exposed port of the Dockerfile
```

Replace `<Image>`, `<Memory>`, `<CPU>`, `<Port>` with the correct values.

What is the value for `<Port>`?

- **`9696`**

Apply this deployment using the appropriate command and get a list of running Pods. 
You can see one running Pod.

```bash
kubectl apply -f deployment.yaml
kubectl get pods
kubectl describe pod <pod-name>
kubectl top pod
```


## Question 7

Let's create a service for this deployment (`service.yaml`):

```yaml
apiVersion: v1
kind: Service
metadata:
  name: <Service name>
spec:
  type: LoadBalancer
  selector:
    app: <???>
  ports:
  - port: 80
    targetPort: <PORT>
```

Fill it in. What do we need to write instead of `<???>`?

- **`9696`**

Apply this config file.


## Testing the service

We can test our service locally by forwarding the port 9696 on our computer 
to the port 80 on the service:

```bash
kubectl port-forward service/<Service name> 9696:80
```

Run `q6_test.py` (from the homework 5) once again to verify that everything is working. 
You should get the same result as in Question 1.

```bash
kubectl apply -f service.yaml
kubectl get services
kubectl port-forward service/subscription 9696:80
python q6_test.py
```

## Autoscaling

Now we're going to use a [HorizontalPodAutoscaler](https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale-walkthrough/) 
(HPA for short) that automatically updates a workload resource (such as our deployment), 
with the aim of automatically scaling the workload to match demand.

Use the following command to create the HPA:

```bash
kubectl autoscale deployment subscription --name subscription-hpa --cpu-percent=20 --min=1 --max=3
```

You can check the current status of the new HPA by running:

```bash
kubectl get hpa
```

The output should be similar to the next:

```bash
NAME               REFERENCE                 TARGETS   MINPODS   MAXPODS   REPLICAS   AGE
subscription-hpa   Deployment/subscription   1%/20%    1         3         1          27s
```

`TARGET` column shows the average CPU consumption across all the Pods controlled by the corresponding deployment.
Current CPU consumption is about 0% as there are no clients sending requests to the server.
> 
>Note: In case the HPA instance doesn't run properly, try to install the latest Metrics Server release 
> from the `components.yaml` manifest:
> ```bash
> kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
>```

```bash
kubectl get deployment metrics-server -n kube-system

kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml

kubectl patch -n kube-system deployment metrics-server --type=json -p '[{"op":"add","path":"/spec/template/spec/containers/0/args/-","value":"--kubelet-insecure-tls"}]'

kubectl get deployment metrics-server -n kube-system
kubectl top pods
kubectl get hpa
```

## Increase the load

Let's see how the autoscaler reacts to increasing the load. To do this, we can slightly modify the existing
`q6_test.py` script by putting the operator that sends the request to the subscription service into a loop.

```python
while True:
    sleep(0.1)
    response = requests.post(url, json=client).json()
    print(response)
```

Now you can run this script.

```bash
rm main.py

uv init
uv add fastapi uvicorn scikit-learn
uv run uvicorn app:app --host 0.0.0.0 --port 8080 --reload

curl http://localhost:8080/health
```

## Question 8 (optional)

Run `kubectl get hpa subscription-hpa --watch` command to monitor how the autoscaler performs. 
Within a minute or so, you should see the higher CPU load; and then - more replicas. 
What was the maximum amount of the replicas during this test?

```bash
uv run uvicorn app:app --host 0.0.0.0 --port 9696 --reload

curl -X POST http://localhost:9696/predict \
  -H "Content-Type: application/json" \
  -d '{"lead_source": "organic_search", "number_of_courses_viewed": 3, "annual_income": 50000}'

python q6_test.py
kubectl get hpa subscription-hpa --watch
```

* 1
* 2
* **`3`**
* 4

> Note: It may take a few minutes to stabilize the number of replicas. Since the amount of load is not controlled 
> in any way it may happen that the final number of replicas will differ from initial.

## Submit the results

* Submit your results here: https://courses.datatalks.club/ml-zoomcamp-2025/homework/hw10
* If your answer doesn't match options exactly, select the closest one. If the answer is exactly in between two options, select the higher value.