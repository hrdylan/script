

if __name__ == "__main__":
    line_gen = LineGenerator("char_lines.csv", "sd_lines.csv", "set_lines.csv")
    cdp = CSVDataProcessor("script_seq_data.csv")
    cdp.partition()
    cdp.examine()
    net = ScriptSeqNet(cdp.X_train, cdp.Y_train, cdp.X_test, cdp.Y_test, 99)
    net.build_model()
    net.train()
    gen = Generator(net, line_gen)
    gen.create(1)
    #gen.eval()
    #gen.display()
    gen.works_to_txt()
    #print(gen.realized_works)
