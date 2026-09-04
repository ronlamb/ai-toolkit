Please summarize each change and place it into the folder called #file:optimization-documentation:  This folder will have the following subfolders:
- files-changed
- performance-change-analysis
- proposed-changes

This first part is just a plan, where you will ask me questions, and we will come up with the details to put in each file.

Once this plan is finalized, you first task will be to  creating prompt files in #file:prompts to run the following tasks, in order

# 1. Populate older files-changed
In the files-changse folder create a single subfolder for eachmodule changed.  In this folder create:
   - `change-summary.md` that contains a summary of all the changes in the file
   - A single file describing each change, before and after.

Create prompt that helps the next step.

# 2. Determine what changes are still valid

Once this is done, analyze the files created in step 1 and dete
If they are not valid move them to a folder called obsolete in #file:optimization-documentation.

Create prompt that helps the next step.

# 3. Populate performance-change-analysis folder

From the remaining optimizations, create a single folder for each individual change defined from step 2.  In each folder:
- Create a summary.md file for the change
- Create an analysis file called `change-analysis.md` that:
   - Critiques the change
   - Finds area of merging with other changes.
   - Finds code improvements to the change.
   - Break the change down separately by
     - CUDA only change
     - MPS only change
     - Generic to both
- Create a pytest unit test to add to #file:tests, but don't add it to tests.

The purpose of this is so that I can go one by one for each change and work on these changes in the base root repository one by one.

Create prompt that helps the next step.

# 4. Populate proposed-changes folder

This is the last step, and is the most important part.  Analyze the information collected in step three and create individual changes, marking whether the change is self contained or dependent.

In the `proposed-changes` folder, create a single folder for each change with descriptions on what to change.
