**21-07-2026**

- [x] handle dates (prevent opening unused time data) and set date range at the end of the file (dates are REQUIRED_YEAR_START and REQUIRED_YEAR_END for ssps, and 1850-2015 for historical)
- [x] make output path configurable in process_combination()
- [x] handle variants: make sure the variant being opened is the one in GCM_REGISTRY
- [x] display a progress bar (using rich library) displaying time that has passed and amount of combinations processed
- [x] implement logging errors (output should be a csv what details which combinations were sucessfully computed, and why the combinations that failed failed). a tthe end of each run of "calculate_ET0" it should display this table using the rich library
- [x] trial run: all models, 1 experiment each     

**22-07-2026**

- [ ] check out trial run errors
- [ ] replace files in daily-ET0 with those in daily-ET0/new
- [ ] commit and push to GitHub 
- [x] remove attribute ET0:standard_name = "air_temperature" 
- [ ] inform hsin. ask which attributes are needed, and show some plots and ask him whether results make sense or not and how i can check everything is correct


**For reproducbility**
- write `environment.yml` file
- check out sign of Rnet (hflss+hfss)
- 
