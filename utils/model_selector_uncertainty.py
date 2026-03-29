class ModelSelector:
        def select_model(self, plan_details):
            """
            Selects the appropriate model based on the classification.

            Parameters:
            - plan_details: Dictionary containing 'classification' and 'recommended_model'.

            Returns:
            - model_type: String identifier of the selected model.
            """
            classification = plan_details.get('classification', 'Unknown')
            recommended_model = plan_details.get('recommended_model', 'Unknown')

            if recommended_model == 'denoising':
                return 'denoising'                                               
            elif recommended_model == 'motion_correction':
                return 'motion_correction'                                                      
            elif classification == 'undersampling':
                return 'undersampling_model'                   
            else:
                return 'Unknown'

