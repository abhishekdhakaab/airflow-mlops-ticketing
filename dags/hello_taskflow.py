from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator

def say_hello():
    print("Airflow is alive. Let's build this thing!")

with DAG(
    dag_id="hello_dag",
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,   # only run when you click it
    catchup=False,
    tags=["setup"],
) as dag:
    hello = PythonOperator(
        task_id="hello_task",
        python_callable=say_hello
    )