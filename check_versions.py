import sys
import pkg_resources
import platform

def check_versions():
    print("Python version:", sys.version)
    print("\nOperating System:", platform.system(), platform.version())
    print("\nChecking required packages:")
    print("-" * 50)
    
    required_packages = [
        'numpy',
        'tensorflow',
        'scikit-learn',
        'Pillow',
        'django',
        'mysqlclient',
        'opencv-python',
        'matplotlib',
        'pandas'
    ]
    
    max_name_length = max(len(name) for name in required_packages)
    
    for package in required_packages:
        try:
            version = pkg_resources.get_distribution(package).version
            status = "✓ Installed"
            print(f"{package:<{max_name_length}} : v{version:<10} : {status}")
        except pkg_resources.DistributionNotFound:
            status = "✗ Not found"
            print(f"{package:<{max_name_length}} : {'N/A':<10} : {status}")
    
    print("\nTensorFlow GPU Support:")
    print("-" * 50)
    try:
        import tensorflow as tf
        print("GPU Available:", tf.test.is_built_with_cuda())
        print("GPU Devices:", tf.config.list_physical_devices('GPU'))
    except ImportError:
        print("TensorFlow not installed")
    except Exception as e:
        print("Error checking GPU:", str(e))

if __name__ == "__main__":
    check_versions() 