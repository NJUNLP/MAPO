# Guide to prepare the preference optimization dataset

1. Run sampling.sh to sample reasoning processes.
2. Run extract_mutliL_en.py to paired En-Answers.
3. Run PreferenceEstimate.sh to Score the output.
4. Run extract_dpo_data.py to get Paired Preference dataset

Please pay attention to the "#SET here" in the files and set the path according to your actual running environment.