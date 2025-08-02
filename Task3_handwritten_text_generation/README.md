# Task 3: Handwritten Text Generation

## 🎯 Objective
To create a tool that converts digital text into a realistic-looking handwritten image. This project explores the practical application of generative AI techniques through a user-friendly script.

## 🛠️ Technology & Approach

This project utilizes the `pywhatkit` library in Python to achieve the text-to-handwriting conversion.

-   **Why `pywhatkit`?** Training a sophisticated generative model (like a GAN or an RNN with attention) from scratch is a highly complex task requiring massive datasets (like the IAM Handwriting Database) and significant computational power. For a project with practical deadlines, leveraging a powerful, pre-existing library is a smart and effective engineering decision. It demonstrates the ability to find and implement the right tool for the job.
-   **How it works:** The library uses pre-built logic and font styles to render the input text onto a blank image, simulating a natural handwriting flow and appearance.

The script is interactive, prompting the user for the text they wish to convert and the desired output filename.

## 🏆 Example Result
When the script is run, it asks for input text. For example, if the user enters "Hello, world! This is my final project for the CODSOFT internship.", the script will generate a PNG image file in the `output/` directory that looks like this text was written by hand on a piece of paper.

## 🚀 How to Run

1.  **Clone the Repository:**
    ```bash
    # This assumes you have already cloned the main CODSOFT repo
    cd CODSOFT/
    ```

2.  **Navigate to the Project Directory:**
    ```bash
    cd Task3_handwritten_text_generation/
    ```

3.  **Install/Update Dependencies:**
    Make sure your virtual environment is activated and you have added `pywhatkit` to your `requirements.txt`. Then run the install command from the **root `CODSOFT` folder**:
    ```bash
    # First, go back to the root folder
    cd .. 
    # Then install
    pip install -r requirements.txt
    ```

4.  **Run the Script:**
    Navigate back into the task folder and run the script.
    ```bash
    cd Task3_handwritten_text_generation/
    python generate_handwriting.py
    ```
    The script will then prompt you to enter your text and a filename. The output image will be saved in the `output/` sub-directory.