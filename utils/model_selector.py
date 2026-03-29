class ModelSelector:
    def __init__(self):

        self.model_mapping = {
            'cyclegan for motion correction': 'motion_correction',
            'cyclegan for mri denoising': 'denoising',
            'u-net based model for mri reconstruction': 'reconstruction',
        }

    def select_model(self, plan_details):
        """
        Determines which domain expert model to use based on the parsed plan.

        Parameters:
        - plan_details: A dictionary containing classification, plan, and recommended_model.

        Returns:
        - The identifier of the selected model.
        """

        recommended_model = plan_details.get('recommended_model')
        if isinstance(recommended_model, str) and recommended_model.lower() != 'unknown':
            recommended_model_lower = recommended_model.lower()
            for key in self.model_mapping:
                if key in recommended_model_lower:
                    return self.model_mapping[key]


        classification = plan_details.get('classification', '').lower()
        if 'motion' or 'artifacts' in classification:
            return 'motion_correction'
        elif 'noise' in classification or 'noisy' in classification:
            return 'denoising'
        elif 'undersampled' in classification or 'undersampling' in classification:
            return 'reconstruction'
        elif 'no correction' in classification or 'no corruption' in classification:
            return 'corruption_free'
        else:
            return None                           
