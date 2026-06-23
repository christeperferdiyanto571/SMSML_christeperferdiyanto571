
from prometheus_client import start_http_server, Gauge
import time
import random

# Metrik untuk dimonitor
MODEL_ACCURACY = Gauge('model_accuracy', 'Current Model Accuracy')

if __name__ == '__main__':
    start_http_server(8001)
    print("Exporter jalan di port 8001")
    while True:
        # Simulasi nilai akurasi
        MODEL_ACCURACY.set(random.uniform(0.8, 0.99))
        time.sleep(5)
