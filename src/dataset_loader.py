import openml
import pandas as pd
import pickle
import os
from typing import Dict, Tuple, Optional
import hashlib

# Loads datasets from OpenML Benchmark Suites and caches them locally
class DatasetLoader:
    def __init__(self, cache_dir: str = "benchmark_cache"):
        # Checks if cache directory exists, creates if not
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    # Function to get datasets from benchmark suite with caching
    def get_benchmark_suite(self, suite_id: int, max_datasets: Optional[int] = None, cache: bool = True, reload: bool = False) -> Dict[str, Tuple[pd.DataFrame, str]]:
        # Generates the name of the cache file
        cache_file = self.generate_cache_filename(suite_id, max_datasets)
        
        # It tries to load from the cache first
        if cache and not reload:
            cached_data = self.load_from_cache(cache_file)
            if cached_data is not None:
                print(f"Loaded {len(cached_data)} datasets from cache: {cache_file}")
                return cached_data
        
        # If its not in the cache, it loads from OpenML
        datasets = self.load_from_openml(suite_id, max_datasets)
        
        # It saves the dataset to cache for future use
        if cache and datasets:
            self.save_to_cache(datasets, cache_file)
            print(f"Cached {len(datasets)} datasets to: {cache_file}")
        
        return datasets
    
    # Function to generate filename for caching
    def generate_cache_filename(self, suite_id: int, max_datasets: Optional[int]) -> str:
        # Generates a unique hash based on the suite ID and number of datasets
        params = f"suite_{suite_id}_max_{max_datasets}"
        hash_id = hashlib.md5(params.encode()).hexdigest()[:8]
        return os.path.join(self.cache_dir, f"benchmark_{hash_id}.pkl")
    
    # Function to load datasets from cache
    def load_from_cache(self, cache_file: str) -> Optional[Dict]:
        # If the file is in the cache, it loads and returns it
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Cache load failed: {e}")
        return None
    
    # Function to save datasets to cache
    def save_to_cache(self, datasets: Dict, cache_file: str):
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(datasets, f)
        except Exception as e:
            print(f"Cache save failed: {e}")

    # Function to load datasets from OpenML
    def load_from_openml(self, suite_id: int, max_datasets: Optional[int]) -> Dict[str, Tuple[pd.DataFrame, str]]:
        try:
            benchmark_suite = openml.study.get_suite(suite_id)
            task_ids = benchmark_suite.tasks
            
            if max_datasets:
                task_ids = task_ids[:max_datasets]
                
            datasets = {}
            successful_loads = 0
            failed_loads = 0
            
            print(f"Loading {len(task_ids)} datasets from OpenML suite {suite_id}...")
            
            for tid in task_ids:
                try:
                    task = openml.tasks.get_task(tid)
                    dataset = openml.datasets.get_dataset(task.dataset_id)
                    name = dataset.name
                    target_attribute = dataset.default_target_attribute
                    
                    if target_attribute is None:
                        print(f"⚠️ Dataset {name} has no default target attribute.")
                        failed_loads += 1
                        continue
                    
                    X, y, _, _ = dataset.get_data(dataset_format="dataframe", target=target_attribute)
                    df = pd.concat([X, y.rename(target_attribute)], axis=1)
                    datasets[name] = (df, target_attribute)
                    successful_loads += 1
                    
                    print(f"Loaded {name}: {df.shape[0]} rows, {df.shape[1]} cols")
                    import time
                    time.sleep(0.5)
                    
                except Exception as e:
                    print(f"Failed to load task {tid}: {e}")
                    failed_loads += 1
                    continue
            
            print(f"Successfully loaded {successful_loads}/{len(task_ids)} datasets "
                  f"({failed_loads} failed)")
            return datasets
        
        except Exception as e:
            print(f"Failed to get benchmark suite {suite_id}: {e}")
            return {}
    
    # Function to clear cache
    def clear_cache(self, suite_id: int = None, max_datasets: int = None):
        if suite_id is None:
            # Clear all cache
            for file in os.listdir(self.cache_dir):
                if file.endswith('.pkl'):
                    os.remove(os.path.join(self.cache_dir, file))
            print("Cleared all cache files")
        else:
            # Clear specific cache
            cache_file = self.generate_cache_filename(suite_id, max_datasets)
            if os.path.exists(cache_file):
                os.remove(cache_file)
                print(f"Cleared cache: {cache_file}")

# Testing the functions above
if __name__ == "__main__":
    loader = DatasetLoader()
    all_datasets = loader.get_benchmark_suite(suite_id=99, max_datasets=10) 
    if all_datasets:
        print(f"Successfully processed {len(all_datasets)} datasets:")
        for name, (df, target) in list(all_datasets.items())[:5]: 
            print(f"{name}: {df.shape} (target: {target})")

        total_rows = sum(df.shape[0] for df, _ in all_datasets.values())
        total_cols = sum(df.shape[1] for df, _ in all_datasets.values())
        print(f"\nTotal: {total_rows:,} rows, {total_cols:,} columns across {len(all_datasets)} datasets")
    else:
        print("Failed to load datasets")
