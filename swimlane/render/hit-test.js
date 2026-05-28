export function binarySearchRange(tasks, timeValue, startAccessor, endAccessor) {
  let lo = 0;
  let hi = tasks.length - 1;
  let match = -1;

  while (lo <= hi) {
    const mid = Math.floor((lo + hi) / 2);
    const task = tasks[mid];
    const start = startAccessor(task);
    const end = endAccessor(task);

    if (timeValue < start) {
      hi = mid - 1;
      continue;
    }
    if (timeValue > end) {
      lo = mid + 1;
      continue;
    }
    match = mid;
    break;
  }

  return match;
}

export function hitTestLane(tasks, timeValue, lineValue, startAccessor, endAccessor, lineAccessor) {
  const index = binarySearchRange(tasks, timeValue, startAccessor, endAccessor);
  if (index < 0) return null;

  const task = tasks[index];
  if (lineAccessor && lineValue != null && lineAccessor(task) !== lineValue) {
    return null;
  }
  return task;
}
