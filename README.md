# GridSearcher 𖣯🔍

---

GridSearcher is a pure Python project designed to simplify the process of running grid searches for Machine Learning 
projects. It serves as a robust alternative to traditional bash scripts, providing a more flexible and user-friendly 
way to manage and execute multiple programs in parallel. 

⚠️ **It is designed for systems where users have direct SSH access 
to machines and can run their python scripts right away.**

# Features ✨󠁇󠁇󠁇
- **Grid Search Made Easy:** Define parameter grids effortlessly and the cartesian product of your hyper-parameters 
will be computed automatically and an instance of your script will be run for all possible combinations.
- **Parallel Execution:** Run multiple programs concurrently, maximizing your computational resources.
- **GPU Scheduling:** Built-in GPU allocation ensures efficient use of available GPUs. Specify the number of GPUs and 
jobs per GPU, and **GridSearcher** will handle the rest
- **Flexible Configuration:** Easily control the number of parallel jobs and GPU assignments through a scheduling 
dictionary.
- **Pure Python:** No more dealing with complex bash scripts. **GridSearcher** is written entirely in Python, making it 
easy to integrate into your existing Python workflows.

# Why GridSearcher? 🤔
- **User-Friendly:** Simplifies the setup and execution of grid searches, allowing you to focus on your Machine 
Learning models.
- **Efficient Resource Management:** Optimize the use of your GPUs and computational resources.
- **Pythonic Approach:** Seamlessly integrates with your Python projects and leverages Python's rich ecosystem.
- **Direct SSH Access:** Ideal for systems where users have direct SSH access to machines, providing a straightforward 
setup and execution process without the need for SLURM or other workload managers, ensuring a smooth and efficient operation.
 
# Installation 🛠️
Install **GridSearcher** via pip:

```shell
pip install gridsearcher
```

# How to use GridSearcher?

---

We provide a minimal working example in the file [example.py](https://github.com/ionutmodo/GridSearcher/blob/main/example.py).


# Contribute 🤝

---

We welcome contributions! If you have suggestions for new features or improvements, feel free to open an issue or submit a 
pull request.
