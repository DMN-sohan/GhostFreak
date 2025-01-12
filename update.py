import argparse
import yaml
import os


def parse_arguments():
    parser = argparse.ArgumentParser(description="Update YAML configuration file")
    parser.add_argument(
        "--args",
        nargs=argparse.REMAINDER,
        help="Arguments to update in format: --key1 value1 --key2 value2",
    )
    return parser.parse_args()


def update_yaml_file(args):
    yaml_path = "cfg/setting.yaml"

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(yaml_path), exist_ok=True)

    # Load existing YAML file if it exists
    if os.path.exists(yaml_path):
        with open(yaml_path, "r") as file:
            config = yaml.safe_load(file) or {}
    else:
        config = {}

    # Process arguments
    if args.args:
        i = 0
        while i < len(args.args):
            arg = args.args[i]
            if arg.startswith("--"):
                key = arg[2:]  # Remove '--' prefix
                if i + 1 < len(args.args) and not args.args[i + 1].startswith("--"):
                    value = args.args[i + 1]

                    # Convert 'true'/'false' to boolean, keep everything else as string
                    if value.lower() == "true":
                        value = True
                    elif value.lower() == "false":
                        value = False

                    # Update config
                    config[key] = value
                    i += 2
                else:
                    print(f"Warning: No value provided for argument {key}")
                    i += 1
            else:
                print(f"Warning: Invalid argument format: {arg}")
                i += 1

    # Write updated config back to file
    with open(yaml_path, "w") as file:
        yaml.dump(config, file, default_flow_style=False)
        print(f"Updated {yaml_path} successfully")


def main():
    args = parse_arguments()
    update_yaml_file(args)


if __name__ == "__main__":
    main()
