#ifndef PRISM_EXTRACTION_HPP
#define PRISM_EXTRACTION_HPP

struct PrismCage;
namespace prism {
  [[deprecated]] bool mid_surface_extraction(PrismCage&);
  bool shell_extraction(PrismCage& pc, bool base);
}

#endif