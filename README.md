This code creates a minimum-length nozzle design using the Method of Characteristics 

when first run set OVERWRITE_IFT to True
if running for the first time (where IFT.json does not exist), if one requires it to run fast, set IFT_ENTRIES to a reasonably small number, i.e. 100

configuration for setup is handled in config.py, i.e. number of Expansion fans

main code, which runs the MoC is in nozzle_design.py

keep codebase in same repository.
