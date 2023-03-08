library(rjson)
library(igraph)
library(RColorBrewer)

##############################################################################
#                             Plot settings                                  #
##############################################################################

# make sure to set the path to the correct json file where you can specify which protocols are present per category
# if you use the default dataset, you can keep the file as is, the correct protocols will be included
categories <- fromJSON(file = "./data/input/plot-protocols.json")

# directory where the plots will be saved
PATH_PLOT_DIR <- "./results/plots/networks"

# format of the plot files choose [png, pdf, svg]
FORMAT_PLOT_FILE <- "pdf"

# create directory for saving plot files
dir.create(PATH_PLOT_DIR)
dir.create(paste0(PATH_PLOT_DIR, "/", FORMAT_PLOT_FILE, "/"))

# size of the plot in inches
size <- 62

vertex_size_multiplicator <- size * 0.12
edge_width_multiplicator <- size * 0.4

# Ratio height:width of the plot (9:16 = 0.5625)
ratio <- 0.707

# size of fonts in plots
cex_size <- size/10

# centrality measure that should be used for node sizes
CENTRALITY_NAME <- "pagerank"

# set node color palette
colorSet1 <- brewer.pal(3, "Set1")
colors <- colorSet1[1:2]
colors <- append(colors, rev(brewer.pal(8,"Greens")))
colors[11] <- "orange"

for (protocols in categories) {
  for (protocol in protocols) {
    print(protocol)
    file <- paste0("./data/processed/defi-networks/", protocol, ".graphml")
  
    g <- read_graph(
      file,
      format = "graphml"
    )
    
    pagerank_ordered <- order(vertex_attr(g, CENTRALITY_NAME), decreasing=TRUE)
    pagerank_sorted <- vertex_attr(g, CENTRALITY_NAME, index=pagerank_ordered)
  
    max_transactions <- max(edge_attr(g, "transaction_count"))
  
    vertex_size <- 1 + (vertex_size_multiplicator * vertex_attr(g, CENTRALITY_NAME))
    edge_width <- (2 + (edge_width_multiplicator * edge_attr(g, "transaction_count")/max_transactions))
    vertex_color <- ifelse(vertex_attr(g, CENTRALITY_NAME) >= pagerank_sorted[5], colors[match(vertex_attr(g, CENTRALITY_NAME), pagerank_sorted)], colors[11])
    
    if (FORMAT_PLOT_FILE=="svg"){
      svg(paste0(PATH_PLOT_DIR, "/svg/", protocol, ".svg"), width = size, height = size * ratio)
    }
      
    if (FORMAT_PLOT_FILE=="png") {
      png(paste0(PATH_PLOT_DIR, "/png/", protocol, ".png"), width = size, height = size * ratio, units="in", res=100)
    }
    
    if (FORMAT_PLOT_FILE=="pdf") {
      pdf(file= paste0(PATH_PLOT_DIR, "/pdf/", protocol, ".pdf"), width = size, height = size * ratio)
    }
    
    layout_algo <- layout_with_kk(g, coords=1.1*layout_in_circle(g))
    
    plot(simplify(g), layout=layout_algo,
                  vertex.color=vertex_color, vertex.size=vertex_size,
                  vertex.label=ifelse(vertex_attr(g, CENTRALITY_NAME) >= vertex_attr(g, CENTRALITY_NAME, index=pagerank_ordered[5]),
                             match(vertex_attr(g, CENTRALITY_NAME), pagerank_sorted), NA),
                  vertex.label.cex=sqrt(vertex_attr(g, CENTRALITY_NAME)) * cex_size, vertex.label.color="white",
                  edge.width=edge_width, edge.color=rgb(190/255,190/255,190/255,0.5) , asp=ratio)
  
    # title(main=protocol, cex.main=cex_size, line=-10)
    
    # new legend for top nodes by pagerank
    sizeCut <- round(pagerank_sorted[1:5], digits=6)
    sizeCutScale <- 1 + (vertex_size_multiplicator * sizeCut)
    
    pagerank_ordered_top <- pagerank_ordered[1:5]
    
    top_node_labels <- ifelse(vertex_attr(g, "etherscan", index=pagerank_ordered_top) != 'no label' & vertex_attr(g, "etherscan", index=pagerank_ordered_top) != "", vertex_attr(g, "etherscan", index=pagerank_ordered_top),
                              ifelse(vertex_attr(g, "label", index=pagerank_ordered_top)=="",'no label',vertex_attr(g, "label", index=pagerank_ordered_top)))

    top_node_legend_placement <- 'bottomright'
    x_sub <- 0

    prot1 <- c('uniswap', 'synthetix', 'yearn')
    if(protocol %in% prot1) {
      top_node_legend_placement <- 'topright'
    }
    prot2 <-  c('barnbridge', 'convex')
    if(protocol %in% prot2) {
      top_node_legend_placement <- 'bottomleft'
      x_sub <- 0.04
    }

    a <- legend(x=top_node_legend_placement,legend=top_node_labels,cex=cex_size, col='white',
                pch=21, pt.bg='white', title="Nodes by PR", y.intersp=2, bty="n")
    x <- (a$text$x + a$rect$left) / 2.05 - x_sub
    y <- a$text$y

    symbols(x,y,circles=sizeCutScale/200,inches=FALSE,add=TRUE,bg=colors[1:10])
  
    # legend_edge_widths <- round(c(min(edge_attr(g, "transaction_count")), mean(edge_attr(g, "transaction_count")) , max_transactions), digits=0)

    # legend('topleft', legend=legend_edge_widths,
    #       col="grey", lwd=(2 + (edge_width_multiplicator *legend_edge_widths/max_transactions)),
    #       cex=cex_size, title="Transactions", bty="n")
  
    dev.off()
  }
}

