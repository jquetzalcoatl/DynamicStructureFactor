include("./util/Corrv3.jl")

parsed_args = parseCommandLine()


dict = init_dict(parsed_args)
main(dict)
