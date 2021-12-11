# Self-attentive classification-based anomaly detection in unstructured logs

This repository is the **unofficial** implementation of [Self-attentive classification-based anomaly detection in unstructured logs](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9338283&casa_token=aXQCQgTo9YYAAAAA:NJJhyLJgWfkeAjPCkMFAJYWmXyqLUk10u4RtYzT6HYVx0M-YZyq5nUU3BwyQt-GBgyTI51WYfGc&tag=1). 

>📋  Please find a demo Colab notebook at the src folder at project root

## Requirements

To install requirements locally and run notebook locally, verify the dependencies in the `requirements.txt`:

```setup
pip install -r requirements.txt
```

When using our implementation demo, simply import the notebook at `src/model/anomaly_detection.ipynb` and modify the folder path to point to your [datasets](https://www.usenix.org/cfdr-data).

Baselines: We implemented two baselines used in the paper - PCA and Deeplog. Please refer to corresponding notebooks for their specifics.
 
## Training

To train the model(s) in the paper, import the notebook with TPU runtime and parallel execution strategy on, each epoch at batch size 512 will take less than 2 mins for first 5 million rows of data.

## Evaluation

The results can be evaluated by observing the F1-score, Recall, Precision and Accuracy. The threshold derivation is automatically iterated and can be observed.

## Results

Please review the results based on our project report [NOT DISCLOSED FOR NOW]. 

Generally we have evidence to prove that the results are reproduciable (also surpassing previous state-of-the-art DeepLog) with some potential evaluation flaws.

## Reproducing Baselines

If you want to run PCA yourself, please: 

```cd baselines/PCA/code```

```python main.py```


If you want to run Deeplog: 

```cd baselines/Deeplog/code```

```python main.py```


## To cite the original paper

<pre><code>@article{nedelkoski2020self,
  title={Self-Attentive Classification-Based Anomaly Detection in Unstructured Logs},
  author={Nedelkoski, Sasho and Bogatinovski, Jasmin and Acker, Alexander and Cardoso, Jorge and Kao, Odej},
  journal={arXiv preprint arXiv:2008.09340},
  year={2020}
}
</code></pre>

## To cite our reproduced work
Please click on the button `Cite this repository` below the repo description. A bibitex will be generated for your convinience.

## License and contributions
This code is released under GPLV3 License.

Pull requests and issues are welcomed to enhance the implementation.

