import ray
from ray.autoscaler.sdk import request_resources


def main() -> None:
    ray.init(address="auto")
    request_resources(bundles=[{"CPU": 8, "GPU": 1, "accelerator_type:T4": 1}])
    print("requested_t4")


if __name__ == "__main__":
    main()
