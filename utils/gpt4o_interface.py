import openai

class GPT4oInterface:
    def __init__(self, client, model):
        """
        Initializes the GPT4o interface with the provided client instance.
        """
        self.model = model
        self.client = client                                                         

    def get_plan(self, image_url, prompt):


        prompt += f"\nHere is the MRI image: {image_url}\n"

        try:

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": image_url},
                            },
                        ],
                    }
                ]
            )


            try:
                gpt_response = response.choices[0].message.content
                return gpt_response
            except AttributeError:
                pass                                             


            try:
                response_dict = response.dict()
                gpt_response = response_dict['choices'][0]['message']['content']
                return gpt_response
            except AttributeError:
                pass                                             


            try:
                gpt_response = response['choices'][0]['message']['content']
                return gpt_response
            except TypeError:
                pass                                        


            response_str = str(response)
            import re
            match = re.search(r"content=\"(.*?)\", refusal=None", response_str)
            if match:
                gpt_response = match.group(1)
                return gpt_response
            else:
                print("Could not extract content from response")
                return None

        except Exception as e:
            print(f"Error communicating with GPT-4o: {e}")
            return None

    def get_initial_classification(self, image_url, prompt):

        prompt += f"\nHere is the MRI image: {image_url}\n"

        try:

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": image_url},
                            },
                        ],
                    }
                ]
            )


            try:
                gpt_response = response.choices[0].message.content
                return gpt_response
            except AttributeError:
                pass                                             


            try:
                response_dict = response.dict()
                gpt_response = response_dict['choices'][0]['message']['content']
                return gpt_response
            except AttributeError:
                pass                                             


            try:
                gpt_response = response['choices'][0]['message']['content']
                return gpt_response
            except TypeError:
                pass                                        


            response_str = str(response)
            import re
            match = re.search(r"content=\"(.*?)\", refusal=None", response_str)
            if match:
                gpt_response = match.group(1)
                return gpt_response
            else:
                print("Could not extract content from response")
                return None

        except Exception as e:
            print(f"Error communicating with GPT-4o: {e}")
            return None


    def get_agent_plan(self, image_url, prompt):


        prompt += f"\nHere is the MRI image: {image_url}\n"

        try:

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": image_url},
                            },
                        ],
                    }
                ]
            )


            try:
                gpt_response = response.choices[0].message.content
                return gpt_response
            except AttributeError:
                pass                                             


            try:
                response_dict = response.dict()
                gpt_response = response_dict['choices'][0]['message']['content']
                return gpt_response
            except AttributeError:
                pass                                             


            try:
                gpt_response = response['choices'][0]['message']['content']
                return gpt_response
            except TypeError:
                pass                                        


            response_str = str(response)
            import re
            match = re.search(r"content=\"(.*?)\", refusal=None", response_str)
            if match:
                gpt_response = match.group(1)
                return gpt_response
            else:
                print("Could not extract content from response")
                return None

        except Exception as e:
            print(f"Error communicating with GPT-4o: {e}")
            return None

