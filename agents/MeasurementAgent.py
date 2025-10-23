from utilities.utility import query_model, parse_big5, persona_card


class MeasurementAgent:
    def __init__(self, scenario, backend_str="gpt4", big5_enabled=False, personality="") -> None:
        # conversation history between doctor and patient
        self.agent_hist = ""
        # presentation information for measurement
        self.presentation = ""
        # language model backend for measurement agent
        self.backend = backend_str
        # prepare initial conditions for LLM
        self.scenario = scenario
        self.big5_enabled = big5_enabled
        self.personality = personality
        self.pipe = None
        self.reset()

    def inference_measurement(self, question) -> str:
        answer = str()
        answer = query_model(self.backend,
                             "\nHere is a history of the dialogue: " + self.agent_hist + "\n Here was the doctor measurement request: " + question,
                             self.system_prompt())
        self.agent_hist += question + "\n\n" + answer + "\n\n"
        return answer

    def system_prompt(self) -> str:
        base = "You are an measurement reader who responds with medical test results. Please respond in the format \"RESULTS: [results here]\""
        if self.big5_enabled:
            measurement_big5 = parse_big5(self.personality)
            base = base + persona_card("Measurement", measurement_big5)
        presentation = "\n\nBelow is all of the information you have. {}. \n\n If the requested results are not in your data then you can respond with NORMAL READINGS.".format(
            self.information)
        return base + presentation

    def add_hist(self, hist_str) -> None:
        self.agent_hist += hist_str + "\n\n"

    def reset(self) -> None:
        self.agent_hist = ""
        self.information = self.scenario.exam_information()