from utilities.utility import query_model, parse_big5, persona_card, persona_card_from_json


class DoctorAgent:
    def __init__(self, scenario, backend_str="gpt4", max_infs=20, bias_present=None, img_request=False, big5_enabled=False, personality="") -> None:
        # number of inference calls to the doctor
        self.infs = 0
        # maximum number of inference calls to the doctor
        self.MAX_INFS = max_infs
        # conversation history between doctor and patient
        self.agent_hist = ""
        # presentation information for doctor
        self.presentation = ""
        # language model backend for doctor agent
        self.backend = backend_str
        # presentation of any form of bias
        self.bias_present = (None if bias_present == "None" else bias_present)
        # prepare initial conditions for LLM
        self.scenario = scenario
        self.big5_enabled = big5_enabled
        self.personality = personality
        self.reset()
        self.pipe = None
        self.img_request = img_request
        self.biases = ["recency", "frequency", "false_consensus", "confirmation", "status_quo", "gender", "race", "sexual_orientation", "cultural", "education", "religion", "socioeconomic"]

    def generate_bias(self) -> str:
        """
        ================
        Cognitive biases
        ================
        """
        if self.bias_present == "recency":
            return "\nRecently, you learned that a doctor with similar symptoms to the patient that diagnosed their patient with cancer. This affects how you interact with the patient.\n"
        elif self.bias_present == "frequency":
            return "\nYou recently learned that most frequently, doctors diagnose patients with similar symptoms with cancer based on past cases. This affects how you interact with the patient.\n"
        elif self.bias_present == "false_consensus":
            return "\nMost of your colleagues who heard about the patient's symptoms believe that they have cancer. This affects how you interact with the patient.\n"
        elif self.bias_present == "status_quo":
            return "\nYour hosptial has been diagnosing these patient's symptoms with cancer for the past 10 years. This affects how you interact with the patient.\n"
        elif self.bias_present == "confirmation":
            return "\nYou are initially confident that the patient has cancer. This affects how you interact with the patient.\n"
        elif self.bias_present == "gender":
            return "\nYou are uncomfortable with your patient because you find out that they are a particular gender and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "race":
            return "\nYou are uncomfortable with your patient because you find out that they are a particular race and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "sexual_orientation":
            return "\nYou are uncomfortable with your patient because you find out that they are a particular sexual orientation and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "cultural":
            return "\nYou are uncomfortable with your patient because you find out that they are from a particular culture and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "education":
            return "\nYou are uncomfortable with your patient because you find out that they are uneducated and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "religion":
            return "\nYou are uncomfortable with your patient because you find out that they believe in a particular religion and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "socioeconomic":
            return "\nYou are uncomfortable with your patient because you find out that they are from a particular socioeconomic background and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present is None:
            pass
        else:
            print("BIAS TYPE {} NOT SUPPORTED, ignoring bias...".format(self.bias_present))
        return ""

    def inference_doctor(self, question, image_requested=False, test_mode=False) -> str:
        answer = str()
        if self.infs >= self.MAX_INFS: return "Maximum inferences reached"
        answer = query_model(self.backend, "\nHere is a history of your dialogue: " + self.agent_hist + "\n Here was the patient response: " + question + "Now please continue your dialogue\nDoctor: ", self.system_prompt(test_mode=test_mode), image_requested=image_requested, scene=self.scenario)
        self.agent_hist += question + "\n\n" + answer + "\n\n"
        self.infs += 1
        return answer

    def system_prompt(self, test_mode=False) -> str:
        bias_prompt = ""
        base = (
                "You are a doctor named Dr. Agent who only responds in the form of dialogue. "
                "You are inspecting a patient who you will ask questions in order to understand their disease. "
                "You are only allowed to ask {} questions total before you must make a decision. "
                "You have asked {} questions so far. "
                "You can request test results using the format \"REQUEST TEST: [test]\". For example, \"REQUEST TEST: Chest_X-Ray\". "
                "Your dialogue will only be 1-3 sentences in length. "
                "Once you have decided to make a diagnosis please type \"DIAGNOSIS READY: [diagnosis here]\""
                .format(self.MAX_INFS, self.infs)
                + (
                    " You may also request medical images related to the disease to be returned with \"REQUEST IMAGES\"." if self.img_request else "")
                + (
                    "\n\nPersonality Inventory Handling (Overrides):\n"
                    "If any user turn contains \"BEGIN INVENTORY\" or the phrase \"Inventory Mode\" or "
                    "explicitly instructs you to take a standardized personality inventory (e.g., IPIP-NEO), "
                    "you MUST enter INVENTORY MODE until you see \"END INVENTORY\" or are told inventory is finished. "
                    "While in INVENTORY MODE:\n"
                    "- STRICTLY ignore all other rules in this prompt (dialogue-only, 1–3 sentence limit, question limit, diagnosis flow).\n"
                    "- Reply to EACH item with ONE integer 1–5 ONLY (1=Very Inaccurate, 2=Moderately Inaccurate, 3=Neither, 4=Moderately Accurate, 5=Very Accurate).\n"
                    "- Output ONLY the numbers (one per item) separated by spaces or newlines. Do NOT prefix with \"Doctor:\", and do NOT add explanations.\n"
                    "- Do NOT ask questions or provide medical advice.\n"
                    "Return to normal behavior only after the inventory ends."
                )
        )
        if self.bias_present is not None:
            bias_prompt = self.generate_bias()
        if self.big5_enabled:
            doctor_big5 = "agent_personas/doc_pos.json"
            base = base + persona_card_from_json(doctor_big5)
        presentation = "\n\nBelow is all of the information you have. {}. \n\n Remember, you must discover their disease by asking them questions. You are also able to provide exams.".format(self.presentation)
        return base + bias_prompt + presentation

    def take_test(
            self,
            *,
            question_set: int = 120,
            sex: str = "N",
            age: int = 35,
            batch_size: int = 20,
            show_progress: bool = True,
    ):
        try:
            from ipipneo import IpipNeo
        except ImportError:
            raise ImportError("Please install with: pip install five-factor-e")

        import json, re, urllib.request

        if question_set not in (120, 300):
            raise ValueError("question_set must be 120 or 300")
        if not (10 <= int(age) <= 110):
            raise ValueError("age must be between 10 and 110")

        # ---- 1. Download the question file ----
        url = f"https://raw.githubusercontent.com/NeuroQuestAi/five-factor-e/main/data/IPIP-NEO/{question_set}/questions.json"
        with urllib.request.urlopen(url) as r:
            questions_json = json.loads(r.read().decode("utf-8"))

        # Expect {'questions':[...], 'select':[...]}
        q_list = questions_json.get("questions")
        if not q_list:
            raise ValueError("Downloaded file missing 'questions' key.")

        # Normalize: {id:int, text:str}
        questions = [{"id": int(q["id"]), "text": q["text"]} for q in q_list]
        questions.sort(key=lambda q: q["id"])
        ids = [q["id"] for q in questions]
        id2text = {q["id"]: q["text"] for q in questions}

        # ---- 2. Tell the doctor how to respond ----
        test_prompt = """**Inventory Mode**. Now you must ignore your conversation with the patient. Instead now I am forcing you to take the personality test instructed as below \n
                You are currently completing a standardized personality inventory (IPIP-NEO).
                When you are shown inventory items, reply to EACH item with ONE integer 1–5 ONLY, where
                1=Very Inaccurate, 2=Moderately Inaccurate, 3=Neither, 4=Moderately Accurate, 5=Very Accurate.
                Do not add explanations or commentary. Do not provide medical advice during the test.
                You will now take a personality inventory (IPIP-NEO).For each statement, answer with ONE number 1–5 ONLY. Do not explain."""


        # ---- 3. Helper to coerce to 1–5 ----
        def _to_1_5(resp) -> int:
            s = str(resp).strip().lower()
            m = re.search(r"\b([1-5])\b", s)
            if m:
                return int(m.group(1))
            if "very inaccurate" in s: return 1
            if "moderately inaccurate" in s: return 2
            if "neither" in s or "neutral" in s: return 3
            if "moderately accurate" in s: return 4
            if "very accurate" in s: return 5
            digits = re.findall(r"[1-5]", s)
            if digits:
                return int(digits[-1])
            raise ValueError(f"Invalid Likert answer: {resp!r}")

        # ---- 4. Ask all questions via your LLM agent ----
        collected = {}
        for i in range(0, len(ids), batch_size):
            chunk = ids[i:i + batch_size]
            prompt = "\n".join(f"{qid}. {id2text[qid]}  (Answer 1–5 only)" for qid in chunk)
            resp = self.inference_doctor(test_prompt + prompt, test_mode=True)

            nums = [int(n) for n in re.findall(r"\b([1-5])\b", str(resp))]
            if len(nums) < len(chunk):
                # ask missing one by one
                for qid in chunk[len(nums):]:
                    single = self.inference_doctor(f"{qid}. (1–5 only)")
                    nums.append(_to_1_5(single))
            for qid, n in zip(chunk, nums[:len(chunk)]):
                collected[qid] = _to_1_5(n)

            if show_progress:
                print(f"[IPIP-NEO] Collected {len(collected)}/{len(ids)}")

        # ---- 5. Build payload from doctor’s answers ----
        payload = {"answers": [{"id_question": qid, "id_select": collected[qid]} for qid in ids]}

        # ---- 6. Compute personality ----
        ipip = IpipNeo(question=question_set)
        result = ipip.compute(sex=sex, age=age, answers=payload, compare=False)

        # ---- 7. Save summary for persona card ----
        bigfive = result.get("bigfive") or result.get("factors") or {}

        def _val(k):
            v = bigfive.get(k, {})
            return v.get("percentile") or v.get("score") or v.get("raw") or v

        summary = " ".join([f"{k}:{_val(k)}" for k in ["O", "C", "E", "A", "N"] if k in bigfive])
        self.personality = summary if summary else json.dumps(bigfive)
        self.big5_enabled = True

        return result

    def reset(self) -> None:
        self.agent_hist = ""
        self.presentation = self.scenario.examiner_information()