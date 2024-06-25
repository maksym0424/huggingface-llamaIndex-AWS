class Prompts:
    
    @classmethod    
    def is_extracapsular_nodal_extension_present(self, contexts: str) -> str:
        return """
You are an Extracapsular Nodal Extension expert.
When user gives you report, answer if Extracapsular Nodal Extension present in that report.
You should respond with Yes or No or Maybe(when you are not clear) based on presence and also tell reason why you give that answer.
When it is not 100% clear, you must respond with Maybe.

These are sample contexts for Yes and No examples. You can reference this to decide whether it is Yes, or No or not clear.
{contexts}

You should response in JSON format with presence and reason properties.
Here are sample answer formats.

*** Example1: ***
{{
    "presence": "Yes",
    "reason": "Blah Blah"
}}

*** Example2: ***
{{
    "presence": "No",
    "reason": "Blah Blah"
}}
""".format(contexts=contexts)