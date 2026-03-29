import re

class PlanParser:
    def __init__(self):
        pass

    def parse_plan(self, gpt_response):
        result = {
            'classification': None,
            'plan': [],
            'recommended_model': None
        }


        classification_patterns = [
            r"Classification:\s*(.*?)[\n\r]",                                    
            r"It appears the image is (.*?)[\.\n]",
            r"The MRI image shows signs of (.*?)[\.\n]",
            r"indicative of (.*?)[\.\n]",
            r"The image appears to have (.*?)[\.\n]",
            r"identified as (.*?)[\.\n]",
            r"the image is affected by (.*?)[\.\n]",
        ]

        for pattern in classification_patterns:
            match = re.search(pattern, gpt_response, re.IGNORECASE)
            if match:
                result['classification'] = match.group(1).strip()
                break
        else:

            classification = self.extract_classification_keywords(gpt_response)
            result['classification'] = classification if classification else 'Unknown'


        model_patterns = {
            r'cyclegan (model )?for motion correction': 'CycleGAN for motion correction',
            r'cyclegan (model )?for mri denoising': 'CycleGAN for MRI denoising',
            r'u-net (based )?(model )?for mri reconstruction': 'U-Net based model for MRI reconstruction',
        }


        for pattern, model_name in model_patterns.items():
            if re.search(pattern, gpt_response, re.IGNORECASE):
                result['recommended_model'] = model_name
                break
        else:

            recommended_model = self.extract_recommended_model_keywords(gpt_response)
            result['recommended_model'] = recommended_model if recommended_model else 'Unknown'


        plan_start = None
        for heading in ["### Correction Plan", "Correction Plan:", "Plan:", "Steps:"]:
            plan_start = gpt_response.find(heading)
            if plan_start != -1:
                break

        if plan_start != -1:
            plan_text = gpt_response[plan_start:]
            plan_lines = plan_text.strip().split('\n')
            for line in plan_lines:
                line = line.strip()
                if re.match(r'^(\d+\.|\-|\*)\s', line):
                    result['plan'].append(line)
                elif result['plan']:
                    result['plan'][-1] += ' ' + line
                elif line.startswith('###') or line == '':
                    break
        else:

            pass

        return result

    def extract_classification_keywords(self, text):
        text = text.lower()
        classification_keywords = {
            'motion corruption': ['motion', 'movement', 'motion artifact', 'blurring', 'blurry', 'distortion'],
            'noise': ['noise', 'noisy', 'grainy', 'speckle', 'artifact'],
            'undersampled': ['undersampled', 'aliasing', 'undersampling', 'low resolution', 'incomplete data'],
            'no corruption': ['clean', 'motion free', 'no motion', 'no noise', 'clean data', 'clean image', 'no distortion', 'no distortion']
        }

        for classification, keywords in classification_keywords.items():
            if any(keyword in text for keyword in keywords):
                return classification
        return None

    def extract_recommended_model_keywords(self, text):
        text = text.lower()
        model_keywords = {
            'CycleGAN for motion correction': ['cyclegan', 'motion correction'],
            'CycleGAN for MRI denoising': ['cyclegan', 'denoising'],
            'U-Net based model for MRI reconstruction': ['u-net', 'reconstruction']
        }
        for model_name, keywords in model_keywords.items():
            if all(keyword in text for keyword in keywords):
                return model_name
        return None
