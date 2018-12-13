Build the image form the Dockerfile
```
docker build mlspec .
```
To run on the Apache dataset and export the results to a results folder in the current working directory
```
docker run -v $(pwd)/results:app/results:Z mlspec Apache
```
