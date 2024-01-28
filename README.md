# Say-I-Dont-Know
## Introduction
The "Say-I-Dont-Know" project primarily investigates whether AI assistants based on large language models can perceive the boundaries of their own knowledge and express this understanding through natural language. 
This repository contains the code, data and model checkpoints for our paper "[Can AI Assistants Know What They Don't Know?](https://arxiv.org/pdf/2401.13275.pdf)".

<div style="text-align: center;">
    <figure>
        <img src="figures/Knowledge_quadrants.png" alt="Knowledge Quadrants" width="400" height="200">
        <figcaption style="text-align: center;">Figure 1: The knowledge quadrants of an AI assistant.</figcaption>
    </figure>
</div>

The AI assistant’s perception of its own knowledge can be represented through knowledge quadrants.
The knowledge quadrant is a partition which can divide the knowledge into four categories: Known Knowns, Known Unknowns, Unknown Knowns and Unknown Unknowns, as shown in Figure 1.
In this project, we develope model-specific Idk ("I don't know") dataset for the AI assistant, and by utilizing this Idk dataset, we aim to align the assistant to refuse answering questions that it does not know and answer questions that it knows.
Consequently, this transforms knowledge from Unknown-Unknowns and Unknown-Knowns to Known-Knowns and Known-Unknowns, thereby enhancing the truthfulness of the AI assistant.

## Open-source List

<!-- ### Idk Datasets -->


## Idk Dataset and Preference Data
<figure>
    <img src="figures/construct_idk_and_preference_data.png" alt="Contstruction_of_Idk_dataset">
    <figcaption style="text-align: center;">Figure 2: Construction of Idk dataset and preference data.</figcaption>
</figure>