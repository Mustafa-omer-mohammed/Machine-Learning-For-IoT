This Project amis to create a Collaborative Inference  where the inference workload is distributed across interconnected devices to improve the prediction accuracy.
Different devices host different inference pipelines, with complexity and accuracy proportional to their hardware resources. 
Collaborative Inference frameworks generally comprise the following components: 
(i) a Fast inference pipeline running on edge 
(ii) a Slow inference pipeline running on the cloud 
(iii) a “success checker” policy to determine whether the Fast inference was “confident” about its prediction or not; if not, run the Slow inference to get the final prediction


![edge cloud](ML4Iots/HW3/Edge-Cloud%20Collaborative%20Inference/edge_cloud.PNG)
